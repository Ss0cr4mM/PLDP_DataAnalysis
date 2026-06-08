import numpy as np
import matplotlib.pyplot as plt

def finder(target_temp, ref_array):
    """Returns the index of the first value in array that is <= target_temp.
    Used to find where one run's starting temperature matches another run's cooling curve.
    Returns -1 if no such value is found."""
    for i in range(len(ref_array)):
        if ref_array[i] <= target_temp:
            return i
    return -1

def max_temperature(array):
    """Returns the array with the maximum first value, which is the maximum temperature of the run, which is the reference for stitching."""
    first_values = []
    for i in range(len(array)):
        first_values.append(array[i][0])
    winner = first_values.index(max(first_values))
    return winner

def stitch_times(time_full, array_material):
    """Reads the index of the maximum temperature in the material array, then finds the corresponding
    time for this index in every time array (offset), then deletes the array of maximum temperature (the reference)
    from the full time array, creating time_ref, then adds the offset to every value in every time in 
    the full time array, creating the stitched time_ref."""
    print(f"Max temperature array index is {max_temperature(array_material)}.")
    ref_run_time = time_full.pop(max_temperature(array_material))
    ref_run_material = array_material.pop(max_temperature(array_material))
    print(f"Reference run time array is {ref_run_time}.")
    
    for i, item in enumerate(array_material):
        idx = finder(item[0], ref_run_material) # This is the index position where the initial tempeture is in the reference
        print(f"Index found for material {item} is {idx}.")
        if idx == -1:
            raise ValueError(f"Could not find temperature {item[0]:.2f} in reference run data.")
        else:
            offset = ref_run_time[idx] # Corresponding time for the temperature found in ref_run_material
            print(f"Offset for material {item} is {offset:.2f}.")
            time_array = np.array(time_full[i]) # Convert the time array of the current run to a numpy array for easier manipulation
            time_full[i] = time_array + offset # Add the offset to every value in the time array of the current run
    return time_full
        

time = [([7,8,9],[4,5,6],[7,8,9]),
        ([12,11,10],[15,14,13],[18,17,16]),
        ([23,23,2],[23,23,4],[67,543,45])  
    ]

a , b, c = zip(time)
print(a)
print(b)
print(c)























# print(max_temperature([[1,2,3], [4,5,6], [7,8,9]]))
# print(stitch_times(time, materials))

# for i in range(len(array_material)):
#         idx = finder(start_temp[i][0], max_temperature(array_material)) #This is the postion from where we need to find
#         if idx == -1:
#             raise ValueError(f"Could not find temperature {start_temp:.2f} in reference run data.")
#         else:
#             index.append(idx)
#     offset = []
#     for i in range(len(index)):
#         offset = time_full[array_material.index(max_temperature(array_material))][idx]
#     time_ref = time_full.remove(time_full[array_material.index(max_temperature(array_material))])
#     for i in range(len(time_ref)):
#         time_ref[i] = time_ref[i] + offset
#     return time_ref