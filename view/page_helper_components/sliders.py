def get_slider_marks(value):
    start, end = value
    num_marks = 6
    
    # Calculate the step size based on the number of desired marks
    step_size = ((end - start) // (num_marks - 1)) // 10 * 10
    
    # Generate evenly spaced marks that are divisible by 10
    marks = {int(start + i * step_size): str(int(start + i * step_size)) for i in range(num_marks - 1)}
    marks[end] = str(end)
    
    return marks