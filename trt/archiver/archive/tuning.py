from archive.common import ArchiveDict, ArchiveObject


class Tuning(ArchiveDict):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @classmethod
    def create(cls, type=str(), **kwargs):
        return Tuning(**kwargs)


if __name__ != '__main__':
    ArchiveObject.register_creator('@tuning', Tuning.create)
