from optigurator import create_app

if __name__ == '__main__':
    app = create_app()
    app.run(ssl_context=("dev-https.crt", "dev-https.key"))
