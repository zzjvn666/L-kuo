首先要写一个驱动张大头二维云台，指哪打哪的接口


generate_motor_commands(x, y)


根据extract_transformed_segments得到的点就可以画图了


```
                for segment in segments:
                    length = len(segment)
                    for i in range(0, length - 1, stride):
                        x, y = segment[i]
                        generate_motor_commands(x, y)
```



yuanshen.py就能画一个原神
