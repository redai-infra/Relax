#define _GNU_SOURCE

#include <arpa/inet.h>
#include <dlfcn.h>
#include <errno.h>
#include <netinet/in.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

typedef int (*bind_fn_t)(int, const struct sockaddr *, socklen_t);

static bind_fn_t real_bind_fn = NULL;
static pthread_mutex_t log_mutex = PTHREAD_MUTEX_INITIALIZER;

static void format_sockaddr(const struct sockaddr *addr, char *host, size_t host_len, int *port) {
    if (addr == NULL) {
        snprintf(host, host_len, "<null>");
        *port = -1;
        return;
    }

    if (addr->sa_family == AF_INET) {
        const struct sockaddr_in *in = (const struct sockaddr_in *)addr;
        inet_ntop(AF_INET, &in->sin_addr, host, host_len);
        *port = ntohs(in->sin_port);
        return;
    }

    if (addr->sa_family == AF_INET6) {
        const struct sockaddr_in6 *in6 = (const struct sockaddr_in6 *)addr;
        inet_ntop(AF_INET6, &in6->sin6_addr, host, host_len);
        *port = ntohs(in6->sin6_port);
        return;
    }

    snprintf(host, host_len, "family=%d", addr->sa_family);
    *port = -1;
}

static void log_bind_result(int fd, const struct sockaddr *addr, int ret, int saved_errno) {
    char host[INET6_ADDRSTRLEN] = {0};
    int port = -1;
    struct sockaddr_storage bound_addr;
    socklen_t bound_len = sizeof(bound_addr);

    if (ret == 0 && getsockname(fd, (struct sockaddr *)&bound_addr, &bound_len) == 0) {
        format_sockaddr((const struct sockaddr *)&bound_addr, host, sizeof(host), &port);
    } else {
        format_sockaddr(addr, host, sizeof(host), &port);
    }

    const char *path = getenv("BINDLOG_PATH");
    FILE *out = stderr;
    if (path != NULL && path[0] != '\0') {
        out = fopen(path, "a");
        if (out == NULL) {
            out = stderr;
        }
    }

    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);

    pthread_mutex_lock(&log_mutex);
    fprintf(
        out,
        "[bindlog] ts=%ld.%09ld pid=%ld tid=%ld fd=%d rc=%d errno=%d addr=%s:%d\n",
        (long)ts.tv_sec,
        ts.tv_nsec,
        (long)getpid(),
        (long)syscall(SYS_gettid),
        fd,
        ret,
        saved_errno,
        host,
        port
    );
    fflush(out);
    pthread_mutex_unlock(&log_mutex);

    if (out != stderr) {
        fclose(out);
    }
}

int bind(int fd, const struct sockaddr *addr, socklen_t len) {
    if (real_bind_fn == NULL) {
        real_bind_fn = (bind_fn_t)dlsym(RTLD_NEXT, "bind");
        if (real_bind_fn == NULL) {
            errno = ENOSYS;
            return -1;
        }
    }

    errno = 0;
    int ret = real_bind_fn(fd, addr, len);
    int saved_errno = errno;
    if (ret == 0 || saved_errno == EADDRINUSE) {
        log_bind_result(fd, addr, ret, saved_errno);
        errno = saved_errno;
    }
    return ret;
}
