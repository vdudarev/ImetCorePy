<path>\.venv\Scripts\activate
pip freeze > requirements.txt
where python
pip list
On the server:
1) Create a virtual environment: 
python.exe -m venv .venv
2) Update pip in the environment:
python.exe -m pip install --upgrade pip
3) recreate all the packages for the environment as described in requirements.txt
pip install -r requirements.txt


Delete virtual environment: 
rmdir /s /q .venv