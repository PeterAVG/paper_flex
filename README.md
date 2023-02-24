# Getting started

### Installing Python 3.11.0
Use pyenv to install python 3.11.0: https://github.com/pyenv/pyenv

### Installing poetry
Poetry is used as a package manager: https://python-poetry.org/docs/#installation

### Setting up the repository and environment
To create the virtual environment with the Python packages run:

    make dev-setup

Source the vitual environment using:

    source .venv/bin/activate

### Running an experiment

Default experiment folder is `src/experiments`. Run the following command to start `TestExperiment` with parameter set (called a partition) as returned by `backfill`.

    python manage.py TestExperiment backfill

The cache will automatically save this particular experiment to a pickle file in `src/experiments/cache`.

NOTE: optional arguments `overwrite` will always overwrite cache with given partition. `append` will return existing experiment if the partition exists in cache already, and otherwise run and save to cache. `dry-run` will run all experiments from partitions, but never save them:

    python manage.py TestExperiment backfill --overwrite
    python manage.py TestExperiment backfill --append
    python manage.py TestExperiment backfill --dry-run

Default behavior is `append`.

You can create multiple partition, i.e., parameter sets, to run new experiments by adding new methods to the class `TestExperiment`:

    python manage.py TestExperiment my_own_append --append

You can also create a new class in a new file in `src/experiments` that inherits the `ETLComponent`:

    @cache
    def some_func(partition: str, **kwargs: Any) -> Any:

        logger.info(partition, kwargs)
        logger.info("Hello world")

        return None

    class MyOwnExperiment(ETLComponent):
    def first_experiment(self, **kwargs: Any) -> None:
        for day in [1,2,3]:
            partition = f"day={day}"
            some_cached_function(partition, **kwargs)

And call it like this:

    python manage.py MyOwnExperiment first_experiment
