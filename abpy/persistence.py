import os
from abpy import datacollection


def write_datacollector_to_csv_file(filename, datacollector, omit_identifier=False):
    ensure_dir(filename)
    fobj = open(filename, "w")
    c = 0
    for d in sorted(datacollector.data.keys()):
        if not omit_identifier:
            fobj.write(str(d) + ";")
        c += 1
        cc = 0
        for v in datacollector.data[d]:
            if isinstance(v, float):
                fobj.write('{:.6f}'.format(v))
            else:
                fobj.write(str(v))
            cc += 1
            if cc < len(datacollector.data[d]):
                fobj.write(";")
        if c < len(datacollector.data):
            fobj.write("\n")


def create_datacollector_from_csv_file(filename, omit_identifier=False):
    fobj = open(filename, "r")

    dc = datacollection.AbstractDataCollector(None, None)

    c = 0
    for line in fobj:
        splitted = line.split(";")
        if omit_identifier:
            dc.data[c] = splitted
        else:
            dc.data[splitted[0]] = list(map(float, splitted[1:]))

    return dc


def write_modelrun_to_csv_files(foldername, modelrun, omit_identifier=False):
    for c in modelrun.collectors:
        write_datacollector_to_csv_file(foldername + "/" + c.name + ".csv", c, omit_identifier)


def write_batch_to_csv_files(foldername, batch, omit_identifier=False):
    c = 0
    for r in batch.runs:
        write_modelrun_to_csv_files(foldername + "/" + str(c) + "/", r, omit_identifier)
        c += 1


def check_dir_exists(d):
    if not len(d) == 0:
        if not os.path.exists(d):
            return False
    return True


def ensure_dir(f):
    d = os.path.dirname(f)
    print(d)
    if not check_dir_exists(d):
        try:
            os.makedirs(d)
        except:
            pass

