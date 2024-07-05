import argparse
import json
from matplotlib import pyplot as plt

def compute_stats(json_data):
    category_count = dict()
    for annotation in json_data["annotations"]:
        if annotation["category_id"] not in category_count:
            category_count[annotation["category_id"]] = 0
        category_count[annotation["category_id"]] += 1

    print(json_data["categories"])
    for category in json_data["categories"]:
        num_occurrences = category_count[category["id"]] if category["id"] in category_count else 0
        print("-%s: %d" % (category["name"], num_occurrences))

    '''
    # compute date distribution by month
    date_count = dict()
    for annotation in json_data["images"]:
        date = annotation["date_captured"]
        month = date.split("-")[1]
        if month not in date_count:
            date_count[month] = 0
        date_count[month] += 1
    # sort and print
    sorted_dates = sorted(date_count.items(), key=lambda x: x[0])
    for date, count in sorted_dates:
        print("%s: %d" % (date, count))
    '''


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    args = parser.parse_args()

    # open json file and modify each annotation by substituting the "file_name" field by removing the substring "data/"
    # e.g. "data/frames/20200514/clip_10_1744/image_0015.jpg" -> "frames/20200514/clip_10_1744/image_0015.jpg"

    with open(args.dataset, "r") as f:
        data = json.load(f)
        for i in range(len(data["images"])):
            data["images"][i]["file_name"] = data["images"][i]["file_name"].replace("data/", "")
        with open(args.dataset, "w") as f:
            json.dump(data, f)


    #dataset = json.load(open(args.dataset, "r"))
    #compute_stats(dataset)
