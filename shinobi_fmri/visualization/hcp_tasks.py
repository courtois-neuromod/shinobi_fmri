"""
HCP task configuration for consistent visualization across the pipeline.

This module centralizes all HCP task-related configuration including icons,
colors, and condition mappings to ensure consistency across all visualizations.
"""

import seaborn as sns

# Define task icons (Unicode symbols that are widely supported)
TASK_ICONS = {
    'Gambling': '♦',    # Diamond
    'Motor': '✱',       # Heavy asterisk
    'Language': '■',    # Black square
    'Social': '★',      # Star
    'Relational': '●',  # Black circle
    'Emotion': '▲'      # Triangle
}

# Define task colors (using seaborn Dark2 palette)
_palette_dark = sns.color_palette("Dark2")
TASK_COLORS = {
    'Gambling': _palette_dark[2],
    'Motor': _palette_dark[3],
    'Language': _palette_dark[4],
    'Social': _palette_dark[5],
    'Relational': _palette_dark[6],
    'Emotion': _palette_dark[7]
}

# Shinobi task color (using Set2 palette)
_palette_set2 = sns.color_palette("Set2")
SHINOBI_COLOR = _palette_set2[1]

# HCP task to conditions mapping
EVENTS_TASK_DICT = {
    'Gambling': ['reward', 'punishment', 'reward-punishment', 'punishment-reward', 'effects_interest'],
    'Motor': ['left_hand', 'right_hand', 'left_foot', 'right_foot', 'tongue', 'cue',
              'left_hand-avg', 'right_hand-avg', 'left_foot-avg', 'right_foot-avg', 'tongue-avg'],
    'Language': ['story', 'math', 'story-math', 'math-story'],
    'Social': ['mental', 'random', 'mental-random'],
    'Relational': ['relational', 'match', 'relational-match'],
    'Emotion': ['face', 'shape', 'face-shape', 'shape-face']
}

# Shinobi conditions (no HCP task association)
SHINOBI_CONDITIONS = ['Kill', 'HIT', 'JUMP', 'HealthLoss', 'DOWN', 'RIGHT', 'LEFT', 'UP', 'Inter']

# Low-level features (Shinobi task sensory/motor confounds)
# These are the internal names as they appear in the data
LOW_LEVEL_CONDITIONS = ['luminance', 'optical_flow', 'audio_envelope', 'button_presses_count']

# Mapping from internal low-level feature names to display names
LOW_LEVEL_DISPLAY_NAMES = {
    'luminance': 'Luminance',
    'optical_flow': 'Optical flow',
    'button_presses_count': 'Button press',
    'audio_envelope': 'Audio envelope'
}

def get_event_to_task_mapping():
    """Get mapping from event/condition name to task name."""
    return {event: task for task, events in EVENTS_TASK_DICT.items() for event in events}

def get_task_colors():
    """Get consistent task colors (for backward compatibility)."""
    return TASK_COLORS.copy()

def get_task_icon(task_name):
    """Get icon for a specific task."""
    return TASK_ICONS.get(task_name, '')

def get_task_color(task_name):
    """Get color for a specific task."""
    return TASK_COLORS.get(task_name, '#808080')  # Default gray if not found

def get_condition_label(condition_name, task_name=None):
    """
    Get formatted condition label with icon if it's an HCP condition.
    Converts low-level internal names to display names.

    Parameters
    ----------
    condition_name : str
        Name of the condition
    task_name : str, optional
        Task name (will be looked up if not provided)

    Returns
    -------
    str
        Formatted label with icon prefix for HCP conditions, or display name for low-level features
    """
    # Convert low-level internal names to display names
    if condition_name in LOW_LEVEL_CONDITIONS:
        return LOW_LEVEL_DISPLAY_NAMES.get(condition_name, condition_name)

    if task_name is None:
        event_to_task = get_event_to_task_mapping()
        task_name = event_to_task.get(condition_name)

    if task_name and task_name in TASK_ICONS:
        icon = TASK_ICONS[task_name]
        return f'{icon} {condition_name}'
    else:
        return condition_name

def get_condition_color(condition_name):
    """
    Get color for a condition (HCP, Shinobi, or low-level feature).

    Parameters
    ----------
    condition_name : str
        Name of the condition

    Returns
    -------
    tuple
        RGB color tuple
    """
    if condition_name in SHINOBI_CONDITIONS or condition_name in LOW_LEVEL_CONDITIONS:
        return SHINOBI_COLOR

    event_to_task = get_event_to_task_mapping()
    task_name = event_to_task.get(condition_name)

    if task_name:
        return TASK_COLORS[task_name]
    else:
        return '#808080'  # Default gray

def get_task_label(task_name):
    """
    Get formatted task label with icon.

    Parameters
    ----------
    task_name : str
        Name of the task

    Returns
    -------
    str
        Formatted label with icon prefix
    """
    icon = TASK_ICONS.get(task_name, '')
    if icon:
        return f'{icon} {task_name}'
    else:
        return task_name

def is_shinobi_condition(condition_name, include_low_level=False):
    """
    Check if a condition is from Shinobi dataset.

    Parameters
    ----------
    condition_name : str
        Name of the condition
    include_low_level : bool
        If True, also consider low-level features as Shinobi conditions

    Returns
    -------
    bool
        True if condition is from Shinobi dataset
    """
    if include_low_level:
        return condition_name in SHINOBI_CONDITIONS or condition_name in LOW_LEVEL_CONDITIONS
    return condition_name in SHINOBI_CONDITIONS

def get_all_shinobi_conditions(include_low_level=False):
    """
    Get list of all Shinobi conditions.

    Parameters
    ----------
    include_low_level : bool
        If True, include low-level features

    Returns
    -------
    list
        List of Shinobi condition names
    """
    if include_low_level:
        return SHINOBI_CONDITIONS + LOW_LEVEL_CONDITIONS
    return SHINOBI_CONDITIONS.copy()
