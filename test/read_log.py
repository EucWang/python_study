import time

filename2 = "data.txt"

def trim_data_to_file():
    filename = "RetTimout_T1_20171228_send.log"

    try:
        f_obj = open(filename)
        f_data = open(filename2, 'w')
        for line in f_obj.readlines():
            # 2017-12-28 00:00:00
            # ,238 INFO AnMb2i :http-apr-8010-exec-27 [com.wzc.common.msg.impl.DispatchCarMessageSender] minaId:hwgw_demo_s1
            # , carid:217361300000032
            # , msg:com.wzc.common.msg.order.app.down.DeleteAllOrderMsg
            items = line.split(',')
            datetime = items[0].strip()
            strptime = time.strptime(datetime, "%Y-%m-%d %H:%M:%S")
            datetime = strptime.tm_hour * 60 * 60 + strptime.tm_min * 60 + strptime.tm_sec

            mina_id = items[1].split(':')[-1].strip()
            if mina_id == 'mina_trunk_demo_s1':
                mina_id = 1
            elif mina_id == 'mina_trunk_demo_s2':
                mina_id = 2
            elif mina_id == 'mina_trunk_demo_s3':
                mina_id = 3
            elif mina_id == 'mina_trunk_demo_s4':
                mina_id = 4
            elif mina_id == 'hwgw_demo_s1':
                mina_id = 5
            elif mina_id == 'hwgw_demo_s2':
                mina_id = 6
            elif mina_id == 'hwgw_1.0.16_product_s3':
                mina_id = 7
            elif mina_id == 'hwgw_demo_s4':
                mina_id = 8

            car_id = items[2].split(':')[-1].strip()

            # device_id = sorted(car_id)
            # print("all requests count: ", len(device_id))
            # device_id = set(device_id)
            # print("all device count : ", len(device_id))

            # for index in range(len(car_id)):
            #     car_id[index] = device_id.index(car_id[index])

            control_type = items[3].split('.')[-1].strip()

            if control_type == 'DeleteAllOrderMsg':
                control_type = 1
            elif control_type == 'OpenDoorMsg':
                control_type = 2
            elif control_type == 'CloseDoorMsg':
                control_type = 3
            elif control_type == 'ReturnCarMsg':
                control_type = 4
            elif control_type == 'UserOrderMsg':
                control_type = 5
            elif control_type == 'RequestUserOrderResultMsg':
                control_type = 6

            write_line = str(datetime) + "\t" + str(mina_id) + "\t" + str(car_id) + "\t" + str(control_type) + "\n"
            f_data.writelines(write_line)

    except BaseException as e:
        print(e)


def load_data():
    data = []
    try:
        with open(filename2) as f:
            for line in f.readlines():
                items = line.strip().split('\t')
                # item = {'datetime': time.strptime(items[0], "%Y-%m-%d %H:%M:%S"), 'mina_id': items[1],
                #         'car_id': items[2], 'control_type': items[3]}
                data.append(items)

    except BaseException as e:
        print(e)
    else:
        return data


def plot_data():
    import matplotlib.pyplot as plt
    data = load_data()

    x = []
    y = []
    z = []
    a = []

    for item in data:
        if len(item) == 4:
            # x.append(time.strptime(item[0], "%Y-%m-%d %H:%M:%S"))
            x.append(float(item[0]))
            y.append(item[1])
            z.append(int(item[2]))
            a.append(item[3])

    print("总数据量", len(data))
    print("len(a)", len(a))

    z_cp = sorted(list(set(z)))
    for index in range(len(z)):
        device_index = z_cp.index(z[index])
        z[index] = device_index

    print('总设备数: z_cp', len(z_cp))
    print('z[0:10', z[0:10])
    print('len(x)', len(x))

    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

    figure = plt.figure(figsize=(50, 30), dpi=160)

    axes = plt.subplot(111)
    print('x[-1]', x[-1])
    # plt.axis([0, int(x[-1]), 0, len(z_cp)])

    type1_x = []
    type1_y = []

    type2_x = []
    type2_y = []

    type3_x = []
    type3_y = []

    type4_x = []
    type4_y = []

    type5_x = []
    type5_y = []

    type6_x = []
    type6_y = []

    max_time = x[-1]
    max_device_id_index = len(z_cp)
    print('z_cp[-1]', len(z_cp))

    for i in range(len(data)):
        if z[i] == 0:
            print("z[i]", i, 'z_cp[%d]' % i, z_cp[0])

        if a[i] == '1':  # DeleteAllOrderMsg
            type1_x.append(x[i]/max_time)
            type1_y.append(z[i]/max_device_id_index)
        elif a[i] == '2':  # OpenDoorMsg
            type2_x.append(x[i]/max_time)
            type2_y.append(z[i]/max_device_id_index)
        elif a[i] == '3':  # CloseDoorMsg
            type3_x.append(x[i]/max_time)
            type3_y.append(z[i]/max_device_id_index)
        elif a[i] == '4':  # ReturnCarMsg
            type4_x.append(x[i]/max_time)
            type4_y.append(z[i]/max_device_id_index)
        elif a[i] == '5':  # UserOrderMsg
            type5_x.append(x[i]/max_time)
            type5_y.append(z[i]/max_device_id_index)
        elif a[i] == '6':  # RequestUserOrderResultMsg
            type6_x.append(x[i]/max_time)
            type6_y.append(z[i]/max_device_id_index)



    print('len(1x, 1y)', len(type1_x), len(type1_y), type1_x[-10:])
    print('len(2x, 2y)', len(type2_x), len(type2_y))
    print('len(3x, 3y)', len(type3_x), len(type3_y))
    print('len(4x, 4y)', len(type4_x), len(type4_y))
    print('len(5x, 5y)', len(type5_x), len(type5_y))
    print('len(6x, 6y)', len(type6_x), len(type6_y))

    type1 = axes.scatter(type1_x, type1_y, s=1, c='red')
    type2 = axes.scatter(type2_x, type2_y, s=1, c='green')
    type3 = axes.scatter(type3_x, type3_y, s=1, c='blue')
    type4 = axes.scatter(type4_x, type4_y, s=1, c='yellow')
    type5 = axes.scatter(type5_x, type5_y, s=1, c='gray')
    type6 = axes.scatter(type6_x, type6_y, s=1, c='#336699')
    axes.legend((type1, type2, type3, type4, type5, type6),
                (u'DeleteAllOrderMsg', u'OpenDoorMsg', u'CloseDoorMsg', u'ReturnCarMsg', u'UserOrderMsg', u'RequestUserOrderResultMsg'),
                loc=2)

    plt.title("时间,设备号,请求类型")
    plt.xlabel("时间")
    plt.ylabel("设备号")
    # axes.xaxis_date()
    # figure.autofmt_xdate()
    plt.show()

# trim_data_to_file()
plot_data()