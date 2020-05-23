配置 user 信息：

```shell
$ git config --global user.name 'strongnine'
$ git config --global user.email 'strongnine@163.com'
```

config 的三个作用域：

```shell
$ git config --global
$ git config --local
# 显示 config 设置
$ git config --list --local
$ git config --list --global
```

**创建仓库**

```shell
$ git init
$ git init ./'git_learning'
# 添加文件
$ git add ./README.md
$ git status
$ git commit -m'Add README.md'
$ git log
```

**往仓库里面添加文件**：参考项目 https://github.com/TTN-js/unforGITtable

```shell
# status 会提示目前的状态，Untracked 代表文件从来没有被管理过
$ git status
# add 可以添加多个文件，例如：
$ git add index.hrml images
# 把变更变成一次正式的提交
$ git commit -m'Add index + logo'
$ git log

# 新用法：把工作区上所有已经被 git 管理的文件一起放到暂存区
# 将文件的修改、文件的删除，添加到暂存区
$ git add -u
```

网友「易风」的补充：

>git add -u：将文件的修改、文件的删除，添加到暂存区。
>
>git add .：将文件的修改，文件的新建，添加到暂存区。
>
>git add -A：将文件的修改，文件的删除，文件的新建，添加到暂存区。
>
>git add -A 相对于 git add -u 命令的优点 ： 可以提交所有被删除、被替换、被修改和新增的文件到数据暂存区，而 git add -u 只能操作跟踪过的文件。git add -A 等同于 git add -all。

**重命名的文件**：

```shell
# 想要重命名一个文件，当你修改了之后用 status 会发现
# 提示你有一个文件被删除，有另一个文件是新的 Untracked files
$ git add
$ git rm

# 可以使用 reset 来对暂存区的文件进行重置
# 注意这是一条相对危险的命令
$ git reset --hard

# 重命名操作可以有直接的方式
$ git mv readme readme.md
```

