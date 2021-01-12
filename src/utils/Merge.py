import os


def CheckSuffix(file, suffix):
    if not suffix or len(suffix) == 0:
        return True
    if not suffix in file or len(suffix) >= len(file):
        return False
    filename, file_extension = os.path.splitext(file)
    return filename[(len(suffix)*-1):] == suffix and file_extension == '.dat'


def GetFilesFromDirectory(directory):
    files = []
    if os.path.isdir(directory):
        # r=root, d=directories, f = files
        for r, d, f in os.walk(directory):
            for file in f:
                files.append(os.path.join(r, file))
        return files
    else:
        raise Exception(f"Directory '{directory}' is not valid.")


def WriteBaseDat(trainingFile, name):
    datastarted = False
    dataCount = 0
    traf = open(trainingFile).readlines()
    with open(name, 'w') as new_file:
        # Start header
        for line in traf:
            if line != '\n':
                newline = line
                if datastarted:
                    break
                if newline.split()[0] == '@data':
                    datastarted = True
                elif newline.split()[0] == '@attribute':
                    newline = newline.replace(
                        'real', 'real ').replace(
                        'REAL', 'REAL ').replace('  ', ' ')
                    if newline.split()[2].lower() in ['numeric', 'real', 'integer', 'string']:
                        newline = newline.split(
                        )[0]+' '+newline.split()[1]+' '+newline.split()[2]+'\n'
                new_file.write(newline)
        for i in range(5):
            datastarted = False
            currentFile = trainingFile.replace(
                '-5-1', f"-5-{str(i+1)}")
            tstf = open(currentFile).readlines()
            for line in tstf:
                if line != '\n':
                    newline = line
                    if datastarted:
                        dataCount += 1
                        new_file.write(newline)
                    elif newline.split()[0] == '@data':
                        datastarted = True
                        continue
    print(f"{name}:[{str(dataCount)}]")


def WriteBaseArff(trainingFile, name):
    datastarted = False
    trainingData = 0
    testData = 0
    traf = open(trainingFile).readlines()
    with open(name, 'w') as new_file:
        # Start header
        for line in traf:
            if line != '\n':
                newline = line
                if datastarted:
                    break
                if newline.split()[0] == '@data':
                    datastarted = True
                if newline.split()[0] == '@input' or newline.split()[0] == '@inputs' or newline.split()[0] == '@output' or newline.split()[0] == '@outputs':
                    continue
                elif newline.split()[0] == '@attribute':
                    newline = newline.replace(
                        'real', 'real ').replace(
                        'REAL', 'REAL ').replace('  ', ' ')
                    if newline.split()[2].lower() in ['numeric', 'real', 'integer', 'string']:
                        newline = newline.split(
                        )[0]+' '+newline.split()[1]+' '+newline.split()[2]+'\n'
                new_file.write(newline)
        # Start data for each file
        for fileType in ['tra', 'tst']:
            for i in range(5):
                datastarted = False
                currentFile = trainingFile.replace(
                    '-5-1tra', f"-5-{str(i+1)}{fileType}")
                tstf = open(currentFile).readlines()
                for line in tstf:
                    if line != '\n':
                        newline = line
                        if datastarted:
                            new_file.write(newline)
                            if fileType == 'tst':
                                testData += 1
                            elif fileType == 'tra':
                                trainingData += 1
                        elif newline.split()[0] == '@data':
                            datastarted = True
                            continue

    print(
        f"{name}:[{str(trainingData)}/{str(testData)}]:{str(100*trainingData/(testData+trainingData))}")


os.chdir("C:\\Users\\jrenewhite\\source\\repos\\PBC4cip\\data")

# print(os.getcwd())
files = []
files = list(set().union(files, list(filter(
    lambda file: CheckSuffix(file, '-5-1tra'), GetFilesFromDirectory(os.getcwd())))))


for file in files:
    currentJoin = os.path.join(
        os.getcwd(), os.path.splitext(os.path.basename(file))[0].replace('-5-1tra', '')+'.arff')
    WriteBaseArff(file, currentJoin)

    currentJoin = os.path.join(
        os.getcwd(), os.path.splitext(os.path.basename(file))[0].replace('-5-1', '')+'.dat')
    WriteBaseDat(file, currentJoin)

    currentJoin = os.path.join(
        os.getcwd(), os.path.splitext(os.path.basename(file))[0].replace('-5-1tra', 'tst')+'.dat')
    WriteBaseDat(file.replace('tra','tst'), currentJoin)
