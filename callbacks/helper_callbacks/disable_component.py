def disable_link(style):
    if style is None:
        style = {}
        
    style['pointer-events'] = 'none'    
    style['color'] = 'gray'
    
    return style

def enable_link(style):
    if style is None:
        style = {}
        
    style['pointer-events'] = 'auto'    
    style['color'] = 'royalblue'
    
    return style