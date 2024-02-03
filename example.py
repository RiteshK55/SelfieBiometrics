from facefiltersense import find, verify, analyze

person1 = 3
person2 = 1
filter1 = 1
filter2 = 9

filepath1 = 'Dataset/Image_Data/{}/{}_Filtered_{}.jpg'.format(
    person1, person1, filter1)
filepath2 = 'Dataset/Image_Data/{}/{}_Filtered_{}.jpg'.format(
    person1, person1, filter2)
filepath3 = 'Dataset/Image_Data/{}/{}_Filtered_{}.jpg'.format(
    person2, person2, filter1)

print('\nOutput of find function for person {} filter {}: '.format(person1, filter1))
df = find(filepath1)
print(df.head())

print('\nOutput of verify function for person {} filter {} AND person {} filter {}: '.format(
    person1, filter1, person1, filter2))
result1 = verify(filepath1, filepath2)
print(result1)

print('\nOutput of verify function for person {} filter {} AND person {} filter {}: '.format(
    person1, filter1, person2, filter1))
result2 = verify(filepath1, filepath3)
print(result2)

print('\nOutput of analyze function for person {} filter {}: '.format(person1, filter1))
result3 = analyze(filepath1)
print(result3)
