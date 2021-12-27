export ZSH="/root/.oh-my-zsh"

ZSH_THEME="Soliah"
EDITOR="/usr/bin/nvim"

plugins=(
    git
    history-substring-search
    zsh-autosuggestions
    zsh-syntax-highlighting
)

source $ZSH/oh-my-zsh.sh

autoload -U compinit && compinit