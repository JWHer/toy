import yaml, git, os

def clone(uri, dir):
    if not os.path.isdir(dir): os.mkdir(dir)
    repo = git.Repo.clone_from(uri, dir, branch='master', **{"no-checkout":True})
    # repo = git.Repo.init(uri)
    # file = repo.git.show()

    # index = git.IndexFile(repo, './MLproject')
    # r = index.checkout('MLproject')

    origin = repo.remote()
    cli = origin.repo.git
    cli.checkout('origin/master', 'MLproject')

if __name__ == '__main__':
    clone('https://github.com/mlflow/mlflow-example.git', './tmp')
    # clone('./tmp')

    with open('../MLproject') as f:
        project = yaml.load(f, Loader=yaml.FullLoader)

    params = project['entry_points']['main']['parameters']
    keys = list(params.keys())
    types = [ v['type'] for v in params.values()]
    default = [ v['default'] for v in params.values()]
    print(', '.join(keys))
    print(', '.join(types))
    print(default)