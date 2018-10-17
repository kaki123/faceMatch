import glob, os

def rename():
    n = 1
    print("ran")
    for filename in glob.glob('grs/*'):
        print(n)
        os.rename(filename, "grs/"+str(n)+".jpg")
        n += 1

def main():
    rename()

if __name__ == "__main__":
    main()
