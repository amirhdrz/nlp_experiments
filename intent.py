"""intent.py - Intent
This is the single source of truth of the intents used
"""

from enum import Enum, unique

# TODO: this is a  really crappy way of doing this
@unique
class Intent(Enum):
    CDF_LAB_AVAILABILITY = 1
    COURSE_AVAILABILITY = 2
    COURSE_BREADTH_REQUIREMENTS = 3
    COURSE_CLASS_SIZE = 4
    COURSE_DESCRIPTION = 5
    COURSE_EXCLUSIONS = 6
    COURSE_INSTRUCTOR = 7
    COURSE_LOCATION = 8
    COURSE_OFFICE_HOURS = 9
    COURSE_PREREQ = 10
    COURSE_TITLE = 11
    COURSE_WAITLIST = 12
    DATE_COURSE_ENROLMENT = 13
    DATE_EVENT = 14
    DATE_EXAM = 15
    FOOD = 16
    LIBRARY_SEARCH = 17
    PARKING = 18
    PLACE_CODE_NAME = 19
    PLACE_HOURS = 20
    PLACE_LOCATION = 21
    PRINTER = 22
    TUITION = 23

