import glob, os

def rename():
    n = 1
    for filename in glob.glob('grs\*'):
        os.rename(filename, str(n)+".jpg")
        n += 1

def main():
    rename()

if __name__ == "__main__":
    main()
