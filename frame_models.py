""" frame_models.py - Contains implemented frames
"""

from collections import namedtuple

from intent import Intent
from entity_recognition import EntityType
from frame import Frame, LambdaFrame, UserFillingFrame
from context import Context

import res.strings as rs

#TODO: each frame needs to be given the entities and the tokens

# Each slot filler should know how to fill the its property

class CourseFrame(Frame):

    class CourseCode(Frame):
        """
        :frame_data type: lower-case string containing course-code
        """

        def _frame_data_filler(self):

            # checks if the course code was already mentioned
            # in the current interaction
            last_conversation = self.context.current_topic.conversation[-1]
            for entity in last_conversation.entities:
                # TODO: should perhaps create a unique entity type coursecode
                # implementation below is brittle
                if entity.type == EntityType.COURSE_CODE:
                    course_code = entity.tokens[0].text_lower
                    return course_code

            # course entity was not found in the context
            # DEBUG only used for debugging
            print(rs.course_code_prompt)
            self.context.current_topic.add_expected_entity(EntityType.COURSE_CODE)
            return None

        def user_message(self):
            raise NotImplementedError()


    def __init__(self, context):
        super().__init__(context, parent_frame=None)

        # Instantiating the slots
        lambda_slot = LambdaFrame.partial_constructor(
            context, parent_frame=self, parent_frame_data=self.frame_data
        )

        self.coursecode = CourseFrame.CourseCode(context, self)
        self.title = lambda_slot('title')
        self.description = lambda_slot('description')
        self.prerequisite = lambda_slot('prerequisite')

    def _frame_data_filler(self):
        coursecode = self.coursecode.frame_data()
        if coursecode:
            # DEBUG code
            return {'description': 'Operating Systems',
                    'title': 'csc369',
                    'prerequistie': 'csc237'}
        else: return None

    def user_message(self):
        raise NotImplementedError()


class BuildingFrame(Frame):

    def __init__(self, context):
        super().__init__(context, parent_frame=None)
        self._id = None
        self._data = None

    @property
    def id(self):
        if self._id:
            return self._id
        else:
            #TODO: get the building id from the user
            pass

    @property
    def data(self):
        if self._data:
            return self._data


FS = namedtuple('FS', ['frame', 'slot'])

mapping = {
    # Intent.COURSE_AVAILABILITY: (CourseFrame, CourseFrame.coursecode),
    # Intent.COURSE_BREADTH_REQUIREMENTS: ,
    # Intent.COURSE_CLASS_SIZE: (CourseFrame, CourseFrame.class_size),
    Intent.COURSE_DESCRIPTION: FS(CourseFrame, 'description')
    # Intent.COURSE_EXCLUSIONS: FS(CourseFrame, CourseFrame.exclusions)
}


def create_frame(intent: Intent, context: Context) -> FS:
    """
    This function serves to link Intents to their respective frames
    """
    fs_tuple = mapping[intent]

    # Instantiates frame and binds it to the context
    frame_instance = fs_tuple.frame(context)
    frame_slot = frame_instance.__getattribute__(fs_tuple.slot)

    return FS(frame_instance, frame_slot)


def intent_to_slot(intent: Intent, frame_instance: Frame) -> FS:
    fs_tuple = mapping[intent]
    frame_slot = frame_instance.__getattribute__(fs_tuple.slot)
    return FS(frame_instance, frame_slot)
