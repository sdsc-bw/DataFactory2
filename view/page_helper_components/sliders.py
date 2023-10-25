def get_slider_marks(value):
    start, end = value
    if end <= 300:
        num_marks = 5
    else:
        num_marks = 6
    
    # Calculate the step size based on the number of desired marks
    step_size = ((end - start) // (num_marks - 1)) // 10 * 10
    
    # Generate evenly spaced marks that are divisible by 10
    marks = {int(start + i * step_size): str(int(start + i * step_size)) for i in range(num_marks - 1)}
    marks[end] = str(end)
    
    return marks

def get_slider_marks_nearest_feature(max_value, min_value=1, value='None'):
    if max_value < 10:
        num_marks = max_value
    else:
        num_marks = 5
    
    # Calculate the step size based on the number of desired marks
    step_size = ((max_value - min_value) // (num_marks - 1))
    
    # Generate evenly spaced marks that are divisible by 10
    marks = {int(min_value + i * step_size): str(int(min_value + i * step_size)) for i in range(num_marks - 1)}
    marks[max_value] = max_value
    marks[max_value + 1] = "None"
    
    return marks