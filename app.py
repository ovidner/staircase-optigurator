import click

from optigurator import create_app


@click.command()
@click.argument("data_dir", default="optigurator_data", type=click.Path(exists=True, allow_dash=False, resolve_path=True))
def main(data_dir):
    app = create_app(data_dir=data_dir)
    app.run(ssl_context=("dev-https.crt", "dev-https.key"))


if __name__ == "__main__":
    main()
