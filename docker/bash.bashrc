# If not running interactively, don't do anything
[ -z "$PS1" ] && return

export TERM=xterm-256color

echo -e "\e[1;36m"
cat<<TF
            ███    ██ ███████ ███    ███  ██████   ██████  ██████
            ████   ██ ██      ████  ████ ██    ██ ██      ██    ██
            ██ ██  ██ █████   ██ ████ ██ ██    ██ ██      ██    ██
            ██  ██ ██ ██      ██  ██  ██ ██    ██ ██      ██    ██
            ██   ████ ███████ ██      ██  ██████   ██████  ██████
TF
echo -e "\e[0;33m"

echo -e "\e[36m"
cat<<TF
                            Neural Motion Synthesis

                  A project by Simon Haag & Paul Fauth-Mayer
TF
echo -en "\e[0;33m"

if [[ $EUID -eq 0 ]]; then
  cat <<WARN
WARNING: You are running this container as root, which can cause new files in
mounted volumes to be created as the root user on your host machine.

To avoid this, run the container by specifying
your user's userid in the Dockerfile
WARN
else
  cat <<EXPL
You are running this container as user with ID $(id -u) and group $(id -g),
which should map to the ID and group for your user on the Docker host. Great!
EXPL
fi

# Turn off colors
echo -e "\e[m"
