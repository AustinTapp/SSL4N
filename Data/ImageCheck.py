from Data import *

if __name__ == "__main__":
    image = NiftiData()
    print(NiftiData.get_sample(image))
    for item in image:
        print(item)
