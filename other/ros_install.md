1. ROS2: Humble
# Fully Supported Ubuntu 22.04.5 LTS (Jammy Jellyfish)
# refer to: https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debs.html

# 1.1. Set locale

locale  # check for UTF-8

sudo apt update && sudo apt install locales
sudo locale-gen en_US en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
export LANG=en_US.UTF-8

locale  # verify settings

# 1.2. Setup Sources

sudo apt install software-properties-common
sudo add-apt-repository universe

sudo apt update && sudo apt install curl -y
export ROS_APT_SOURCE_VERSION=$(curl -s https://api.github.com/repos/ros-infrastructure/ros-apt-source/releases/latest | grep -F "tag_name" | awk -F\" '{print $4}')
curl -L -o /tmp/ros2-apt-source.deb "https://github.com/ros-infrastructure/ros-apt-source/releases/download/${ROS_APT_SOURCE_VERSION}/ros2-apt-source_${ROS_APT_SOURCE_VERSION}.$(. /etc/os-release && echo ${UBUNTU_CODENAME:-${VERSION_CODENAME}})_all.deb"
sudo dpkg -i /tmp/ros2-apt-source.deb

# 1.3. Install ROS 2 packages

sudo apt update
sudo apt upgrade
sudo apt install ros-humble-ros-base
# sudo apt install ros-humble-desktop

# Development tools: Compilers and other tools to build ROS packages
sudo apt install ros-dev-tools

# 1.4. Environment setup

source /opt/ros/humble/setup.bash

# 1.5. Uninstall

sudo apt remove ~nros-humble-* && sudo apt autoremove
sudo apt remove ros2-apt-source
sudo apt update
sudo apt autoremove
sudo apt upgrade # Consider upgrading for packages previously shadowed.

2. ROS2: Jazzy
# Fully Supported Ubuntu 24.04.3 (Noble Numbat)
# refer to: https://docs.ros.org/en/jazzy/Installation/Ubuntu-Install-Debs.html

# 2.1. Set locale

locale  # check for UTF-8

sudo apt update && sudo apt install locales
sudo locale-gen en_US en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
export LANG=en_US.UTF-8

locale  # verify settings

# 2.2. Enable required repositories

sudo apt install software-properties-common
sudo add-apt-repository universe

sudo apt update && sudo apt install curl -y
export ROS_APT_SOURCE_VERSION=$(curl -s https://api.github.com/repos/ros-infrastructure/ros-apt-source/releases/latest | grep -F "tag_name" | awk -F\" '{print $4}')
curl -L -o /tmp/ros2-apt-source.deb "https://github.com/ros-infrastructure/ros-apt-source/releases/download/${ROS_APT_SOURCE_VERSION}/ros2-apt-source_${ROS_APT_SOURCE_VERSION}.$(. /etc/os-release && echo ${UBUNTU_CODENAME:-${VERSION_CODENAME}})_all.deb"
sudo dpkg -i /tmp/ros2-apt-source.deb

# 2.3. Install development tools (optional)

sudo apt update && sudo apt install ros-dev-tools

# 2.4. Install ROS 2

sudo apt update
sudo apt upgrade
sudo apt install ros-jazzy-ros-base
# sudo apt install ros-jazzy-desktop

# 2.5. Setup environment

source /opt/ros/jazzy/setup.bash

# 2.6. Uninstall

sudo apt remove ~nros-jazzy-* && sudo apt autoremove
sudo apt remove ros2-apt-source
sudo apt update
sudo apt autoremove
sudo apt upgrade # Consider upgrading for packages previously shadowed.

3. ROS1: Noetic
# Supported Ubuntu 18.04 LTS (Bionic Beaver)
# refer to: https://wiki.ros.org/noetic/Installation/Ubuntu

# 3.1. Setup your sources.list

sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'

# 3.2. Set up your keys

sudo apt install curl # if you haven't already installed curl
curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add -

# 3.3. Installation

sudo apt update

sudo apt install ros-noetic-ros-base
# sudo apt install ros-noetic-desktop
# sudo apt install ros-noetic-desktop-full

# 3.4. Environment setup

echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc
source ~/.bashrc

# 3.5 Dependencies for building packages

sudo apt install python3-rosdep python3-rosinstall python3-rosinstall-generator python3-wstool build-essential
