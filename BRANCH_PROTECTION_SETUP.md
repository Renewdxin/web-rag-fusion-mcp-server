# GitHub 分支保护设置指南

## 🛡️ 分支保护概述

为了保护重要分支免受意外修改，需要设置分支保护规则。

## 📋 设置步骤

### 1. 访问仓库设置

1. 打开你的GitHub仓库: https://github.com/Renewdxin/mcp
2. 点击 **Settings** 标签
3. 在左侧菜单中找到 **Branches**

### 2. 添加分支保护规则

点击 **Add rule** 按钮，为每个重要分支创建保护规则。

## 🔒 Main 分支保护设置

### 分支名称模式
```
main
```

### 推荐保护设置
- ✅ **Require a pull request before merging**
  - ✅ Require approvals: `1`
  - ✅ Dismiss stale PR approvals when new commits are pushed
  - ✅ Require review from CODEOWNERS

- ✅ **Require status checks to pass before merging**
  - ✅ Require branches to be up to date before merging
  - 选择状态检查: `test` (来自你的GitHub Actions)

- ✅ **Require conversation resolution before merging**

- ✅ **Require signed commits**

- ✅ **Require linear history**

- ✅ **Restrict pushes that create files**

- ✅ **Do not allow bypassing the above settings**

- ❌ **Allow force pushes** (不勾选)
- ❌ **Allow deletions** (不勾选)

### 访问限制
- ✅ **Restrict who can push to matching branches**
  - 只允许仓库管理员直接推送

## 🏷️ Release 分支保护设置

### 分支名称模式
```
release/*
```

### 推荐保护设置
- ✅ **Require a pull request before merging**
  - ✅ Require approvals: `1`
  - ✅ Dismiss stale PR approvals when new commits are pushed

- ✅ **Require status checks to pass before merging**
  - ✅ Require branches to be up to date before merging
  - 选择状态检查: `test`

- ✅ **Require conversation resolution before merging**

- ✅ **Require signed commits**

- ✅ **Restrict pushes that create files**

- ✅ **Do not allow bypassing the above settings**

- ❌ **Allow force pushes** (不勾选，除非你是管理员)
- ❌ **Allow deletions** (不勾选)

### 访问限制
- ✅ **Restrict who can push to matching branches**
  - 只允许仓库管理员和维护者

## 🔐 附加安全设置

### 3. 仓库权限设置

在 **Settings** → **Manage access** 中：

1. **Base permissions**: 设置为 `Read`
2. **只邀请必要的协作者**
3. **为不同角色设置适当权限**:
   - **Admin**: 你自己
   - **Maintain**: 核心维护者
   - **Write**: 受信任的贡献者
   - **Triage**: 问题管理者
   - **Read**: 其他协作者

### 4. 安全设置

在 **Settings** → **Security** → **Code security and analysis** 中启用：

- ✅ **Dependency graph**
- ✅ **Dependabot alerts**
- ✅ **Dependabot security updates**
- ✅ **Dependabot version updates**
- ✅ **Code scanning alerts**
- ✅ **Secret scanning alerts**

### 5. Actions 权限

在 **Settings** → **Actions** → **General** 中：

- 选择 **Allow enterprise, and select non-enterprise, actions and reusable workflows**
- ✅ **Allow actions created by GitHub**
- ✅ **Allow actions by Marketplace verified creators**
- 添加允许的actions: `actions/*`, `github/*`

## 📝 CODEOWNERS 文件

创建 `.github/CODEOWNERS` 文件来指定代码审查者：

```
# Global owners
* @Renewdxin

# Core modules
/src/ @Renewdxin
/config/ @Renewdxin

# Documentation
*.md @Renewdxin
/docs/ @Renewdxin

# Configuration files
.github/ @Renewdxin
docker-compose.yml @Renewdxin
requirements.txt @Renewdxin
```

## 🚫 禁止的操作

设置完成后，以下操作将被禁止：

1. **直接推送到 main/release 分支**
2. **强制推送到受保护分支**
3. **删除受保护分支**
4. **绕过Pull Request要求**
5. **未经审核的代码合并**

## ✅ 允许的工作流程

1. **功能开发**:
   ```bash
   git checkout -b feature/new-feature
   # 开发代码
   git push origin feature/new-feature
   # 创建Pull Request到main
   ```

2. **发布流程**:
   ```bash
   git checkout -b release/v0.2.0
   # 准备发布
   git push origin release/v0.2.0
   # 创建Pull Request到main
   ```

3. **紧急修复**:
   ```bash
   git checkout -b hotfix/critical-fix
   # 修复问题
   git push origin hotfix/critical-fix
   # 创建Pull Request到main和release
   ```

## 🔧 验证设置

设置完成后，尝试：

1. 直接推送到main分支 (应该被拒绝)
2. 创建Pull Request (应该正常工作)
3. 合并PR时检查状态检查 (应该要求通过CI)

## 📞 紧急情况

如果需要紧急修改受保护分支：

1. 作为管理员，可以临时禁用保护规则
2. 进行必要修改
3. 重新启用保护规则

**注意**: 仅在真正紧急情况下使用此方法！

---

设置完成后，你的仓库将受到严格保护，确保代码质量和稳定性！