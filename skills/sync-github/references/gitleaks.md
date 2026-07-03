# Gitleaks 安装参考

当 `gitleaks` 命令不存在，或 `pre-commit run gitleaks --all-files` 找不到对应 hook 时，使用本参考。安装前先告诉用户将执行的命令，并等待确认。

## macOS

优先使用 Homebrew：

```bash
brew install gitleaks
gitleaks version || gitleaks --version
```

## Docker 备选

不能安装本机二进制时，可以用官方镜像扫描当前仓库：

```bash
docker run --rm -v "$PWD:/repo" -w /repo ghcr.io/gitleaks/gitleaks:latest dir . --log-level warning --report-format csv --report-path -
```

## Go 源码构建备选

只有在已安装 Go 且用户同意时使用：

```bash
git clone https://github.com/gitleaks/gitleaks.git /tmp/gitleaks
cd /tmp/gitleaks
make build
```

## 验证

安装后回到目标仓库，重新运行同步 skill 要求的安全检查：

```bash
pre-commit run gitleaks --all-files || gitleaks dir . --log-level warning --report-format csv --report-path -
```

参考来源：Gitleaks 官方 README 与 Homebrew formula。
