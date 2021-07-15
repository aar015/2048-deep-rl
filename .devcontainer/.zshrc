export ZSH="/root/.oh-my-zsh"

ZSH_THEME="Soliah"
EDITOR="/usr/bin/nvim"

plugins=(
    git
    zsh-autosuggestions
    zsh-history-substring-search
    zsh-syntax-highlighting
)

source $ZSH/oh-my-zsh.sh

autoload -U compinit && compinit