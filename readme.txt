  ____                                    _                
 / ___|_ __ ___   ___ ___ _ __ _   _     / \   _ __  _ __  
| |  _| '__/ _ \ / __/ _ \ '__| | | |   / _ \ | '_ \| '_ \ 
| |_| | | | (_) | (_|  __/ |  | |_| |  / ___ \| |_) | |_) |
 \____|_|  \___/ \___\___|_|   \__, | /_/   \_\ .__/| .__/ 
                               |___/          |_|   |_|   

This project holds three tasks.

Task 1: A file that catagorises grocery items to their type (banana -> produce) 

Task 2: A file that recommends an extra grocery item based on the user-inputted item (user input of egg will recommend butter as an additional ingredient)

Task 3: A file that takes in an image of a grocery item, identifies which grocery item it is and which category it belongs to.

Requirements:

-------------

The whole folder, including the images, data and tasks.

Libraries including NLTK, NumPy, Keras, Pandas, scikit-learn, AST, collections, Matplotlib and TensorFlow.

Access to task 1's data url

Installation:

-------------

Install the zip folder at GitHub: https://git.arts.ac.uk/23008862/Machine-Learning

=============================================================================================================
TASK ONE 
--------
Data processing
---------------
The database of choice was Amir Mohseni's Grocery List found at ("/datasets/AmirMohseni/GroceryList/data.csv") for its simple, brand-free grocery data that already had the categories. Unfortunately, the data was too sparse for perceptron. To address this, I decided to extend the database by adding additional entries such as "chicken thigh" and "cider". In addition to this, descriptors were added to pre-existing items and made their own item, such as "organic" + "apple" = "organic apple". To streamline the descriptor process, the class ItemExpander was made to modify all the items exactly once and add them to a new dataframe, skipping lines once it got to the bottom where entries like "organic" + "apple" already had one descriptor. Unfortunately, there was a bug in the code that allowed for multiple descriptors to be added, but after looking through the data, all the additional descriptor made sense for the item such as "locally grown" + "organic" + "apple". This increases the item number from 180 to 690 with 64-68 items per category.

Tokenisation and Vectorisation
------------------------------
The goal pipeline for this process became: user input -> tokeniser -> Bag of Words vector. To input the data into the perceptron, the data must first be tokenised to split all the words so that it can then be vectorised, turning each entry into a list ([0, 0, 0, 1]) for it to be able to be inputted in the perceptron and identifiable after the processing. NLTK's word_tokenize was used for its intuitive, one-line tokenisation. A set was then built to hold the items tokens where it then vectorises through the function vector_create utilising Bag of Words representation. Because of the short grocery item names, the Bag of Words vectors are highly sparse, but luckily, this does not negatively affect the perceptron's performance. 

Perceptron
-----------
User input is processed using tokenisation and BoW vectorisation constructed during training to input into the One-vs-Rest perceptron. The One-vs-Rest perceptron was chosen because of it ease in handling catagorical data, with its input, the perceptron asks each catagory if the input belongs there untill one predicts that it does belong. When evaluating the perceptron, there seemed to be a problem: the perceptron was predicting categories incorrectly with a 76% accuracy. After a tutorial with the lecturer, it was decided to implement a prediction confidence threshold. If the prediction was under 50% (0.5), then the category prediction becomes "uncategorised", after implementing this, the perceptron started to predict correctly with a higher accuracy of 97%.

TASK 2
------
Data processing
----------------
After importing the 13k-recipes.csv, I had to reduce each recipe down to only the pure ingredients, without any measurements and descriptors. This began by singling out the "Cleaned_Ingredients" column in the database. Originally I was going through the filtering route using regex to remove patterns in the strings, but then I turned to transforming the strings as the identity of the ingredient needed to be preserved. This was highlighted when it came to a serious issue. First, I was tokenising the whole list (looked like ["2", "large", "egg", "whites", "1", "pound", "new", "potatoes"]), but this led to problems after cleaning, as the identity of the ingredient were lost. When rejoining the recipe as it now looked like (['egg whites potatoes salt black pepper rosemary thyme parsley']), which isn't ideal since I need to keep the individual ingredients intact (["egg whites", "potatoe", "black pepper"....]). 

To combat this, I came back to the tokenising line (tokens = [word_tokenize(str) for str in recipe_df]) and changed it to better retain each ingredient to feed into the Markov Chain. Furthermore, I had to make sure that the recipes were nested in lists properly, as this allowed for ingredient identification after tokenisation.

Next, I went through 30 different recipes and added anything that wasn't an ingredient into a value called descriptor. I then applied the descriptor to the recipes, removing any strings that were also found in the descriptor and appended it to a new list. Afterwards, I reconnected seperated tokens together such as "egg" + "white" = "egg white".

Markov Chain
-------------
I utilised a first-order Markov chain for this task as it depends on prediction through co-occurrence to answer which food should be recommended to the user. The role of this markov chain is to look at all the pair items in the recipe such as eggs & potatoes, and count the total occurances of that combination. A dictionary for transitions was created to count the co-occurrence relationships between grocery items in shopping baskets, allowing the system to recommend additional items based on how frequently products appear together. Afterwards, a conditional probabilities formula was used to convert the numbers into probabilities; this was to estimate the likelihood of an additional item given the current contents of a shopping basket.

TASK 3
-------
Data processing
----------------
Data processing utilised keras's "keras.utils.image_dataset_from_directory" to create a dataset of images as it works well with the keras based model. The dataset from (https://github.com/marcusklasson/GroceryStoreDataset/tree/master/dataset/iconic-images-and-descriptions) included vegtables, packaged and vegtables. For a wide range of categorical representation in the data, the choice was made to pick carrots, tomatoes, yoghurt and bananas, as these all had similar images total (43-45 each) and represented all of the categories in the database. Furthermore, all of them had the image size of 348x348 pixels; these were perfect as they are considered fixed tensors that could be used by the Keras model with no problems. To visualise the data, Matplotlib was utilised to show the pictures with their labels; 0 is the banana, 1 is carrots, 2 is tomatoes, and 3 is yoghurt.

The Model
-----------
For the model, the goal was to make it simple, hierarchical and overfit resistant, keeping it lightweight to match the limited data set size. This began by inputting the layer and rescaling the images to binary, improving numerical stability and convergence speed. The convolution block learns the images edge, corners and gradients. The second convolution is at a higher level, learning more complex features and textures present in the images. The data then gets flattened into a flat vector, ready to be transferred to the dense layers. Softmax is used for this model as it turns raw scores into probability distribution, perfect for our food categorising task.

Catagorising
-------------
To achieve this, task 1's predictions must aid the task. After colecting the test data's prediction, the prediction is tokenised, vectorised and fed through the One-vs-Rest perceptron.
