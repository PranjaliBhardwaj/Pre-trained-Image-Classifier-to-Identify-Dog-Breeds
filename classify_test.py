import pandas as pd

def load_dataset(pet_images):
    dataset = pd.read_csv(dataset_csv)
    return list(zip(dataset['image_path'], dataset['dog']))


start_time = time()


sleep(75)


end_time = time()


tot_time = end_time - start_time


print("\nTotal Elapsed Runtime:", tot_time, "in seconds.")

print("\nTotal Elapsed Runtime:", str( int( (tot_time / 3600) ) ) + ":" +
          str( int(  ( (tot_time % 3600) / 60 )  ) ) + ":" + 
          str( int(  ( (tot_time % 3600) % 60 ) ) ) )

parser = argparse.ArgumentParser()


parser.add_argument('--dir', type = str, default = 'pet_images/', 
                    help = 'path to the folder of pet images')

in_args = parser.parse_args()


print("Argument 1:", in_args.dir)
python check_images.py
python check_images.py --dir pet_images/
## Sets pet_image variable to a filename 
pet_image = "Boston_terrier_02259.jpg"


low_pet_image = pet_image.lower()


word_list_pet_image = low_pet_image.split("_")


pet_name = ""


for word in word_list_pet_image:
    if word.isalpha():
        pet_name += word + " "


pet_name = pet_name.strip()


print("\nFilename=", pet_image, "   Label=", pet_name)

filenames = ["Beagle_01141.jpg", "Beagle_01125.jpg", "skunk_029.jpg" ]
pet_labels = ["beagle", "beagle", "skunk"]
classifier_labels = ["walker hound, walker foxhound", "beagle",
                     "skunk, polecat, wood pussy"]
pet_label_is_dog = [1, 1, 0]
classifier_label_is_dog = [1, 1, 0]


results_dic = dict()
    

for idx in range (0, len(filenames), 1):
       if filenames[idx] not in results_dic:
        results_dic[filenames[idx]] = [ pet_labels[idx], classifier_labels[idx] ]
 
    if pet_labels[idx] in classifier_labels[idx]:
        results_dic[filenames[idx]].append(1)
            

    else:
        results_dic[filenames[idx]].append(0)

for idx in range (0, len(filenames), 1):
    results_dic[filenames[idx]].extend([pet_label_is_dog[idx], 
                                       classifier_label_is_dog[idx]])
        
for key in results_dic:
    print("\nFilename=", key, "\npet_image Label=", results_dic[key][0],
          "\nClassifier Label=", results_dic[key][1], "\nmatch=",
          results_dic[key][2], "\nImage is dog=", results_dic[key][3],
          "\nClassifier is dog=", results_dic[key][4])                        

  
    if sum(results_dic[key][2:]) == 3:
        print("*Breed Match*")
    if sum(results_dic[key][3:]) == 2:
        print("*Is-a-Dog Match*")
    if sum(results_dic[key][3:]) == 0 and results_dic[key][2] == 1:
        print("*NOT-a-Dog Match*")

python check_images.py --dir pet_images/ --arch resnet  --dogfile dognames.txt
     > resnet_pet-images.txt
python check_images.py --dir pet_images/ --arch alexnet  --dogfile dognames.txt  
     > alexnet_pet-images.txt
python check_images.py --dir pet_images/ --arch vgg  --dogfile dognames.txt 
     > vgg_pet-images.txt
