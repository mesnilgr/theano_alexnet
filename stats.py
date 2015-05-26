import json
import pdb
from collections import Counter
import cPickle

"""
json example:

{u'category': u'pants', u'subcategory': u'straight-leg pants', u'color':
u'white', u'gender': u'M', u'link_id': 26545780, u'image-path':
u'/mnt/bigdisk/data/eddie/gender/images/0499/3a5ca9ae-f995-11e4-9ebe-0cc47a1906b8.jpg',
u'short_description': u'Casual Trouser', u'type': u'A', u'long_description': u'
Cotton Twill, Solid Colour, Mid Rise, Skinny, Tapered Leg, Logo Detail, Button,
Zip, Multipockets, Chinos. 98% Cotton, 2% Elastane.', u'product_id': 49707124}
"""

caption = open("captions.txt", "w")
filename = "/mnt/bigdisk/data/eddie/gender/gender_image_paths.json"
fail, nlines = 0, 0
labels = Counter()
with open(filename, 'r') as f:
    for i, line in enumerate(f):
        try:
            row = json.loads(line)
        except:
            fail += 1
        label = "-|-".join([row["gender"], row["type"], row["category"], row["subcategory"]])        
        labels[label] += 1
        nlines += 1
        try:
            caption.write(row["long_description"].encode("UTF-8") + "\n")
        except TypeError:
            print row["long_description"]
            pdb.set_trace()
caption.close()

if fail > 0:
    print "Failed to parse %i rows" % fail
print "Parsed %i rows, Found %i categories" % (nlines, len(labels))

mini = 1200 + 100 + 50 # imagenet baseline 1200, 100, 50 images per train, test, valid
for i in range(1):
    mini_train = [k for k, v in labels.iteritems() if v > mini]
    print "%i categories with at least %i images in the training set" % (len(mini_train), mini)
    mini *= 0.8

cPickle.dump(mini_train, open("/mnt/bigdisk/data/datasets/categories-completed.pkl", "w"))
cPickle.load(open("/mnt/bigdisk/data/datasets/categories-completed.pkl"))
