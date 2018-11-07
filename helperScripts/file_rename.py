import glob, os

def rename():
    n = 1
    print("ran")
    for filename in glob.glob('prof_dodds/*'):
        print(n)
        os.rename(filename, "prof_dodds/prof_dodds"+str(n)+".jpg")
        n += 1

def main():
    rename()

if __name__ == "__main__":
    main()
