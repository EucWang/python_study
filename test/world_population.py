import json

# from pygal.i18n import COUNTRIES
import pygal
from pygal.maps.world import COUNTRIES
from pygal.style import RotateStyle, LightColorizedStyle
from pygal_maps_world.i18n import ASIA


def get_country_code(country_name):
    '''获取2个字母的国家编码'''
    for code, name in COUNTRIES.items():
        if name == country_name:
            return code

filename = '../population_data.json'
try:
    with open(filename) as f:
        json_data = json.load(f)

        cc_populations = {}
        for pop_dict in json_data:
            #{
           #     "Country Name": "Arab World",
           #     "Country Code": "ARB",
           #     "Year": "1960",
           #     "Value": "96388069"
           # },
            if pop_dict['Year'] == '2010':
                country = pop_dict['Country Name']
                population = int(float(pop_dict['Value']))
                #country_code = pop_dict['Country Code']

                code = get_country_code((country))
                if code:
                    cc_populations[code] = population
                #print(code, '\t', country, '\t', str(population))

        pops_1, pops_2, pops_3 = {}, {}, {}

        for code, pop in cc_populations.items():
            if pop < 10000000:
                pops_1[code] = pop
            elif pop < 100000000:
                pops_2[code] = pop
            else:
                pops_3[code] = pop

        world_map = pygal.maps.world.World()
        world_map.title = '世界地图'
        world_map.add('0-10m', pops_1)
        world_map.add('10m-1bn', pops_2)
        world_map.add('>1bn', pops_3)

        # world_map.add('2010', cc_populations)
        world_map.style = RotateStyle('#336699', base_style=LightColorizedStyle)  #
        world_map.render_to_file('map.svg')
except BaseException as e:
    print(e)