Xvfb :99 -screen 0 1900x1080x8 &
export DISPLAY=:99
export VIRTUAL_SCREEN_PREPARED=true
mvn clean test