/*
 * Mach-O stub for PTSX.app. Launch Services rejects a shell script as
 * CFBundleExecutable (kLSNoExecutableErr).
 *
 * python.org's interpreter re-execs into Python.app. If this stub execs that
 * interpreter in-process, macOS treats PTSX.app as having quit. Fork so this
 * process stays the bundle executable and the child can become Python.
 */
#include <CoreFoundation/CoreFoundation.h>
#include <limits.h>
#include <mach-o/dyld.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

static void die_dialog(const char *msg) {
    char cmd[1024];
    snprintf(cmd, sizeof(cmd),
             "/usr/bin/osascript -e 'display dialog \"%s\" buttons {\"OK\"} "
             "default button 1 with icon stop with title \"PTSX\"'",
             msg);
    system(cmd);
    exit(1);
}

static int is_executable(const char *path) {
    struct stat st;
    return stat(path, &st) == 0 && S_ISREG(st.st_mode) && access(path, X_OK) == 0;
}

int main(void) {
    char exe[PATH_MAX];
    uint32_t n = sizeof(exe);
    if (_NSGetExecutablePath(exe, &n) != 0) {
        die_dialog("Could not locate PTSX.");
    }
    char resolved[PATH_MAX];
    if (realpath(exe, resolved) == NULL) {
        die_dialog("Could not resolve PTSX path.");
    }

    /* <root>/PTSX.app/Contents/MacOS/PTSX → <root> (four components). */
    char root[PATH_MAX];
    strncpy(root, resolved, sizeof(root) - 1);
    root[sizeof(root) - 1] = '\0';
    for (int i = 0; i < 4; i++) {
        char *slash = strrchr(root, '/');
        if (slash == NULL || slash == root) {
            die_dialog("PTSX.app is not in the ptsx folder.");
        }
        *slash = '\0';
    }

    char py[PATH_MAX];
    snprintf(py, sizeof(py), "%s/.venv/bin/python3", root);
    if (!is_executable(py)) {
        die_dialog("Run install.sh in the ptsx folder first, then double-click PTSX again.");
    }
    if (chdir(root) != 0) {
        die_dialog("Could not open the ptsx folder.");
    }

    char venv[PATH_MAX];
    snprintf(venv, sizeof(venv), "%s/.venv", root);
    setenv("VIRTUAL_ENV", venv, 1);
    setenv("__PYVENV_LAUNCHER__", py, 1);
    setenv("TK_SILENCE_DEPRECATION", "1", 0);

    const char *old_path = getenv("PATH");
    char pathbuf[PATH_MAX * 2];
    snprintf(pathbuf, sizeof(pathbuf),
             "%s/.venv/bin:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin%s%s",
             root, old_path ? ":" : "", old_path ? old_path : "");
    setenv("PATH", pathbuf, 1);

    pid_t pid = fork();
    if (pid < 0) {
        die_dialog("Could not start PTSX.");
    }
    if (pid == 0) {
        char *args[] = {py, "-m", "ptsx", "app", NULL};
        execv(py, args);
        _exit(127);
    }

    int status = 0;
    while (1) {
        pid_t r = waitpid(pid, &status, WNOHANG);
        if (r == pid) {
            break;
        }
        if (r < 0) {
            die_dialog("PTSX stopped unexpectedly.");
        }
        CFRunLoopRunInMode(kCFRunLoopDefaultMode, 0.25, false);
    }
    if (WIFEXITED(status) && WEXITSTATUS(status) == 127) {
        die_dialog("Could not start PTSX.");
    }
    if (WIFEXITED(status)) {
        return WEXITSTATUS(status);
    }
    return 1;
}
