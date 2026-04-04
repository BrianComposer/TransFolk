from muspy.datasets import EssenFolkSongDatabase

dataset = EssenFolkSongDatabase(
    root=r"D:\BackUpDrive\Programacion\Python\TransFolk\data\essenABC",
    download_and_extract=True,
    overwrite=False,
    cleanup=False,
    verbose=True,
)