# https://github.com/kunwargupta/Python-Practice-Questions/blob/main/Python%20Practice%20Questions.ipynb
# https://github.com/sauravyadav7721/Python_Practice_Questions_for_Data_Science---1000_Challenges_by_Topic/blob/main/Question/01_Basics_Problem_Set.ipynb
list_of_questions = [
    ("Write a program to return whether a given value is above or below 2000. Return \"Above\" or \"Below\"", "assert f(1200) == \"Below\"\nassert f(3400) == \"Above\""),
    ("Count how many sales are above 2500 in a list using a for loop. Return count", "assert f([1200,3400,1500,800,2100,3000,2700,900]) == 3"),
    ("Use if-elif-else to classify a given value as “Low (<1500)”, “Medium (1500–2500)”, or “High (>2500)", "assert f(1500) == \"Medium (1500–2500)\"\nassert f(800) == \"Low (<1500)\"assert f(3000) == \"High (>2500)\""),
    ("Using a while loop, add cumulative sum to a list until the cumulative sum exceeds 10,000 and return the list", "assert f([1200,3400,1500,800,2100,3000,2700,900]) == [1200,4600,6100,6900,9000,12000]"),
    ("Use break to stop iteration once a sale greater than 3000 is found is a given list, and return the index and value", "assert f([1200,3400,1500,800,2100,3000,2700,900]) == (1,3400)"),
    ("Input a number and check whether it's even or odd using modulo and conditional logic. Return \"Odd\' or \"Even\"", "assert f(178) == \"Even\"\nassert f(19803) == \"Odd\""),
    ("Input a number and find the product of its digits", "assert f(362) == 36\nassert f(2782379) == 42336\nassert f(2801) == 0"),
    ("Input total amount, number of people, and return how much each person should pay after rounding off to 1 decimal place", "assert f(2180,28) == 77.9\nassert f(21797,49) == 444.8"),
    ("Input a password. Check if it meets rules like: ≥8 characters, includes a number, and no spaces (using len, strip, type). Return boolean", "assert f(\"2whoh2o3h3ou\") == True\nassert f(\"9j9j\") == False\nassert f(\"hue 9h i\") == False\nassert f(\"shihdihek\") == False"),
    ("Input a string and check whether it's a palindrome (same forward and backward). Return boolean", "assert f(\"abba\") == True\nassert f(\"abcba\") == True\nf(\"abcd\") == False"),

]