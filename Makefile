.DEFAULT_GOAL=build

ssl:
	mkdir ssl && openssl req -x509 -nodes -days 365 -newkey rsa:2048 -keyout ssl/mykey.key -out ssl/mycert.pem

build:
	docker build . -t 2048-jupyter:latest