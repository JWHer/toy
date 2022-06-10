import zipfile, io

def zip():
    file = io.BytesIO()
    return zipfile.ZipFile(file, 'w')

def unzip(z:zipfile.ZipFile):
    zipinfos = z.infolist()
    ret = []
    for zipinfo in zipinfos:
        with z.open(zipinfo) as file:
            ret.append(file.read())
    return ret

if __name__ == '__main__':
    file = io.BytesIO()
    z = zipfile.ZipFile(file, 'w')
    z.writestr('tmp_filename1', b'abc')
    z.writestr('tmp_filename2', b'def')
    b = unzip(z)
    print(b)