import json
import sys

if __name__ == '__main__':
    f_json = sys.argv[1]
    output_file = f_json.split('.')[0] + '_dynamic.json'
    f_json = open(f_json)
    konb_template = json.load(f_json)
    out_knob = {}
    for key in konb_template.keys():
        if konb_template[key]['dynamic'] == "Yes":
            out_knob[key] = konb_template[key]


    with open(output_file, 'w') as fp:
        json.dump(out_knob, fp, indent=4)



