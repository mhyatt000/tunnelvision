virtualenv .venv;
deactivate &> /dev/null; source ./.venv/bin/activate;

pip install -r yolo_requirements.txt
