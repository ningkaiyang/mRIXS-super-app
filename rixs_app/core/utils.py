import re

def natural_sort(file_list: list[str]) -> list[str]:
    """
    Sort a list of strings using natural alphanumeric sorting in-place.

    Mathematics & Performance Context:
    - Natural Ordering: Standard lexicographical sorting orders strings as `["frame_1", "frame_10", "frame_2"]`. 
      Natural sort parses digit groups as numerical values, resulting in `["frame_1", "frame_2", "frame_10"]`.
      This matches filename conventions of experimental frames.
    - Security (DoS Mitigation): Splits filenames into alpha and numeric parts using regular expressions. 
      To prevent Denial of Service (DoS) attacks from extremely long digit strings (which could trigger 
      high CPU utilization or value limits during string-to-integer conversion), numerical digit sequences 
      are truncated to the first 4000 characters before parsing.

    Args:
        file_list: List of string file paths/names.

    Returns:
        list[str]: The naturally sorted list of strings.

    Raises:
        TypeError: If the input is not a list of strings or if non-string values are present.
    """
    if not isinstance(file_list, list):
        raise TypeError("Input must be a list of strings")
    
    for item in file_list:
        if not isinstance(item, str):
            raise TypeError("All elements in file_list must be strings")
            
    def key_func(s: str) -> list:
        return [int(text[:4000]) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]
        
    file_list.sort(key=key_func)
    return file_list
