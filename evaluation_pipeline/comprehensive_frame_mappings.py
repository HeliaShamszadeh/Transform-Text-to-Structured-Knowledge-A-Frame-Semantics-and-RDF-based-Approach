#!/usr/bin/env python3
"""
Comprehensive frame mappings for 50+ frames with 200+ semantic roles.
This provides generic support for any frame type without hard-coding specific inputs.
"""

def get_comprehensive_frame_mappings():
    """Get comprehensive frame-specific and generic role mappings."""
    
    # Frame-specific mappings (50+ frames)
    frame_specific = {
        # Life events
        'Being_born': {'Child': 'has_person', 'Place': 'has_location', 'Time': 'has_time'},
        'Death': {'Deceased': 'has_person', 'Place': 'has_location', 'Time': 'has_time'},
        'Marriage': {'Spouse': 'has_spouse', 'Place': 'has_location', 'Time': 'has_time'},
        'Divorce': {'Spouse': 'has_spouse', 'Place': 'has_location', 'Time': 'has_time'},
        
        # Education and work
        'Education': {'Student': 'has_student', 'Institution': 'has_institution', 'Time': 'has_time'},
        'Employment': {'Employee': 'has_employee', 'Employer': 'has_employer', 'Time': 'has_time'},
        'Retirement': {'Person': 'has_person', 'Time': 'has_time'},
        
        # Achievements and recognition
        'Win_prize': {'Competitor': 'has_competitor', 'Prize': 'has_prize', 'Competition': 'has_competition'},
        'Award': {'Recipient': 'has_recipient', 'Award': 'has_award', 'Time': 'has_time'},
        'Achievement': {'Achiever': 'has_achiever', 'Achievement': 'has_achievement', 'Time': 'has_time'},
        
        # Leadership and authority
        'Leadership': {'Leader': 'has_leader', 'Theme': 'has_theme', 'Time': 'has_time'},
        'Authority': {'Authority': 'has_authority', 'Subject': 'has_subject', 'Time': 'has_time'},
        'Governance': {'Governor': 'has_governor', 'Subject': 'has_subject', 'Time': 'has_time'},
        
        # Movement and location
        'Motion': {'Theme': 'has_theme', 'Destination': 'has_destination', 'Source': 'has_source'},
        'Travel': {'Traveler': 'has_traveler', 'Destination': 'has_destination', 'Time': 'has_time'},
        'Arrival': {'Arriver': 'has_arriver', 'Destination': 'has_destination', 'Time': 'has_time'},
        'Departure': {'Departure': 'has_departure', 'Destination': 'has_destination', 'Time': 'has_time'},
        
        # Communication
        'Communication': {'Speaker': 'has_speaker', 'Addressee': 'has_addressee', 'Message': 'has_message'},
        'Speaking': {'Speaker': 'has_speaker', 'Addressee': 'has_addressee', 'Topic': 'has_topic'},
        'Writing': {'Author': 'has_author', 'Text': 'has_text', 'Time': 'has_time'},
        'Reading': {'Reader': 'has_reader', 'Text': 'has_text', 'Time': 'has_time'},
        
        # Social interactions
        'Meeting': {'Participant': 'has_participant', 'Place': 'has_location', 'Time': 'has_time'},
        'Social_event': {'Participant': 'has_participant', 'Place': 'has_location', 'Time': 'has_time'},
        'Friendship': {'Friend': 'has_friend', 'Time': 'has_time'},
        'Relationship': {'Partner': 'has_partner', 'Time': 'has_time'},
        
        # Conflict and competition
        'Conflict': {'Participant': 'has_participant', 'Issue': 'has_issue', 'Time': 'has_time'},
        'Competition': {'Competitor': 'has_competitor', 'Competition': 'has_competition', 'Time': 'has_time'},
        'War': {'Participant': 'has_participant', 'Location': 'has_location', 'Time': 'has_time'},
        'Battle': {'Participant': 'has_participant', 'Location': 'has_location', 'Time': 'has_time'},
        
        # Creation and production
        'Creation': {'Creator': 'has_creator', 'Created_entity': 'has_created_entity', 'Time': 'has_time'},
        'Production': {'Producer': 'has_producer', 'Product': 'has_product', 'Time': 'has_time'},
        'Manufacturing': {'Manufacturer': 'has_manufacturer', 'Product': 'has_product', 'Time': 'has_time'},
        
        # Consumption and use
        'Consumption': {'Consumer': 'has_consumer', 'Consumed_entity': 'has_consumed_entity', 'Time': 'has_time'},
        'Use': {'User': 'has_user', 'Used_entity': 'has_used_entity', 'Time': 'has_time'},
        'Purchase': {'Buyer': 'has_buyer', 'Seller': 'has_seller', 'Goods': 'has_goods', 'Time': 'has_time'},
        'Sale': {'Seller': 'has_seller', 'Buyer': 'has_buyer', 'Goods': 'has_goods', 'Time': 'has_time'},
        
        # Possession and ownership
        'Possession': {'Possessor': 'has_possessor', 'Possessed_entity': 'has_possessed_entity', 'Time': 'has_time'},
        'Ownership': {'Owner': 'has_owner', 'Owned_entity': 'has_owned_entity', 'Time': 'has_time'},
        'Transfer': {'Donor': 'has_donor', 'Recipient': 'has_recipient', 'Transferred_entity': 'has_transferred_entity', 'Time': 'has_time'},
        
        # Perception and cognition
        'Perception': {'Perceiver': 'has_perceiver', 'Perceived_entity': 'has_perceived_entity', 'Time': 'has_time'},
        'Seeing': {'Seer': 'has_seer', 'Seen_entity': 'has_seen_entity', 'Time': 'has_time'},
        'Hearing': {'Hearer': 'has_hearer', 'Heard_entity': 'has_heard_entity', 'Time': 'has_time'},
        'Thinking': {'Thinker': 'has_thinker', 'Thought': 'has_thought', 'Time': 'has_time'},
        'Belief': {'Believer': 'has_believer', 'Belief': 'has_belief', 'Time': 'has_time'},
        'Knowledge': {'Knower': 'has_knower', 'Known_entity': 'has_known_entity', 'Time': 'has_time'},
        
        # Emotion and attitude
        'Emotion': {'Experiencer': 'has_experiencer', 'Emotion': 'has_emotion', 'Time': 'has_time'},
        'Love': {'Lover': 'has_lover', 'Loved_entity': 'has_loved_entity', 'Time': 'has_time'},
        'Hate': {'Hater': 'has_hater', 'Hated_entity': 'has_hated_entity', 'Time': 'has_time'},
        'Fear': {'Experiencer': 'has_experiencer', 'Feared_entity': 'has_feared_entity', 'Time': 'has_time'},
        'Hope': {'Hoper': 'has_hoper', 'Hoped_for': 'has_hoped_for', 'Time': 'has_time'},
        
        # Judgment and evaluation
        'Judgment': {'Judge': 'has_judge', 'Evaluee': 'has_evaluee', 'Judgment': 'has_judgment', 'Time': 'has_time'},
        'Evaluation': {'Evaluator': 'has_evaluator', 'Evaluee': 'has_evaluee', 'Evaluation': 'has_evaluation', 'Time': 'has_time'},
        'Assessment': {'Assessor': 'has_assessor', 'Assessed_entity': 'has_assessed_entity', 'Assessment': 'has_assessment', 'Time': 'has_time'},
        
        # Categorization and classification
        'Categorization': {'Item': 'has_item', 'Category': 'has_category', 'Time': 'has_time'},
        'Classification': {'Classifier': 'has_classifier', 'Classified_entity': 'has_classified_entity', 'Class': 'has_class', 'Time': 'has_time'},
        'Typing': {'Typer': 'has_typer', 'Typed_entity': 'has_typed_entity', 'Type': 'has_type', 'Time': 'has_time'},
        
        # Temporal and spatial
        'Temporal_collocation': {'Theme': 'has_theme', 'Time': 'has_time'},
        'Spatial_collocation': {'Theme': 'has_theme', 'Location': 'has_location'},
        'Time_vector': {'Theme': 'has_theme', 'Time': 'has_time'},
        'Location': {'Theme': 'has_theme', 'Place': 'has_location'},
        
        # Organization and structure
        'Organization': {'Organization': 'has_organization', 'Member': 'has_member', 'Time': 'has_time'},
        'Membership': {'Member': 'has_member', 'Organization': 'has_organization', 'Time': 'has_time'},
        'Structure': {'Component': 'has_component', 'Whole': 'has_whole', 'Time': 'has_time'},
        
        # Cause and effect
        'Causation': {'Cause': 'has_cause', 'Effect': 'has_effect', 'Time': 'has_time'},
        'Result': {'Cause': 'has_cause', 'Result': 'has_result', 'Time': 'has_time'},
        'Consequence': {'Cause': 'has_cause', 'Consequence': 'has_consequence', 'Time': 'has_time'},
        
        # Change and transformation
        'Change': {'Entity': 'has_entity', 'Attribute': 'has_attribute', 'Time': 'has_time'},
        'Becoming': {'Entity': 'has_entity', 'Attribute': 'has_attribute', 'Time': 'has_time'},
        'Transformation': {'Entity': 'has_entity', 'From_state': 'has_from_state', 'To_state': 'has_to_state', 'Time': 'has_time'},
        
        # Finish and completion
        'Finish_competition': {'Competitor': 'has_competitor', 'Competition': 'has_competition', 'Time': 'has_time'},
        'Completion': {'Completer': 'has_completer', 'Completed_entity': 'has_completed_entity', 'Time': 'has_time'},
        'End': {'Entity': 'has_entity', 'Time': 'has_time'},
        
        # Start and beginning
        'Start': {'Entity': 'has_entity', 'Time': 'has_time'},
        'Beginning': {'Entity': 'has_entity', 'Time': 'has_time'},
        'Initiation': {'Initiator': 'has_initiator', 'Initiated_entity': 'has_initiated_entity', 'Time': 'has_time'},
        
        # Quantification and measurement
        'Quantified_mass': {'Quantity': 'has_quantity', 'Theme': 'has_theme', 'Time': 'has_time'},
        'Measurement': {'Measurer': 'has_measurer', 'Measured_entity': 'has_measured_entity', 'Measurement': 'has_measurement', 'Time': 'has_time'},
        'Counting': {'Counter': 'has_counter', 'Counted_entity': 'has_counted_entity', 'Count': 'has_count', 'Time': 'has_time'},
        
        # Political and legal
        'Political_locales': {'Theme': 'has_theme', 'Location': 'has_location', 'Time': 'has_time'},
        'Legality': {'Entity': 'has_entity', 'Legal_status': 'has_legal_status', 'Time': 'has_time'},
        'Law': {'Lawmaker': 'has_lawmaker', 'Law': 'has_law', 'Time': 'has_time'},
        
        # Health and medical
        'Health': {'Person': 'has_person', 'Health_status': 'has_health_status', 'Time': 'has_time'},
        'Illness': {'Patient': 'has_patient', 'Illness': 'has_illness', 'Time': 'has_time'},
        'Treatment': {'Patient': 'has_patient', 'Treatment': 'has_treatment', 'Time': 'has_time'},
        
        # Education and learning
        'Learning': {'Learner': 'has_learner', 'Learned_entity': 'has_learned_entity', 'Time': 'has_time'},
        'Teaching': {'Teacher': 'has_teacher', 'Student': 'has_student', 'Subject': 'has_subject', 'Time': 'has_time'},
        'Training': {'Trainer': 'has_trainer', 'Trainee': 'has_trainee', 'Skill': 'has_skill', 'Time': 'has_time'},
        
        # Art and creativity
        'Art': {'Artist': 'has_artist', 'Artwork': 'has_artwork', 'Time': 'has_time'},
        'Performance': {'Performer': 'has_performer', 'Audience': 'has_audience', 'Time': 'has_time'},
        'Entertainment': {'Entertainer': 'has_entertainer', 'Audience': 'has_audience', 'Time': 'has_time'},
        
        # Science and research
        'Research': {'Researcher': 'has_researcher', 'Subject': 'has_subject', 'Time': 'has_time'},
        'Discovery': {'Discoverer': 'has_discoverer', 'Discovered_entity': 'has_discovered_entity', 'Time': 'has_time'},
        'Experiment': {'Experimenter': 'has_experimenter', 'Subject': 'has_subject', 'Time': 'has_time'},
        
        # Technology and innovation
        'Innovation': {'Innovator': 'has_innovator', 'Innovation': 'has_innovation', 'Time': 'has_time'},
        'Invention': {'Inventor': 'has_inventor', 'Invention': 'has_invention', 'Time': 'has_time'},
        'Technology': {'User': 'has_user', 'Technology': 'has_technology', 'Time': 'has_time'}
    }
    
    # Generic mappings (200+ roles)
    generic_mapping = {
        # Core semantic roles
        'Agent': 'has_agent', 'Theme': 'has_theme', 'Time': 'has_time', 'Place': 'has_location',
        'Location': 'has_location', 'Person': 'has_person', 'Entity': 'has_entity',
        
        # Participants
        'Participant': 'has_participant', 'Actor': 'has_actor', 'Doer': 'has_doer',
        'Performer': 'has_performer', 'Speaker': 'has_speaker', 'Author': 'has_author',
        'Creator': 'has_creator', 'Producer': 'has_producer', 'Maker': 'has_maker',
        'Builder': 'has_builder', 'Designer': 'has_designer', 'Inventor': 'has_inventor',
        'Discoverer': 'has_discoverer', 'Researcher': 'has_researcher', 'Scientist': 'has_scientist',
        'Artist': 'has_artist', 'Writer': 'has_writer', 'Poet': 'has_poet', 'Novelist': 'has_novelist',
        'Musician': 'has_musician', 'Singer': 'has_singer', 'Dancer': 'has_dancer',
        'Director': 'has_director',
        
        # Recipients and targets
        'Recipient': 'has_recipient', 'Addressee': 'has_addressee', 'Audience': 'has_audience',
        'Viewer': 'has_viewer', 'Listener': 'has_listener', 'Reader': 'has_reader',
        'Student': 'has_student', 'Learner': 'has_learner', 'Trainee': 'has_trainee',
        'Patient': 'has_patient', 'Customer': 'has_customer', 'Buyer': 'has_buyer',
        'User': 'has_user', 'Consumer': 'has_consumer', 'Owner': 'has_owner',
        'Possessor': 'has_possessor', 'Holder': 'has_holder',
        
        # Objects and entities
        'Object': 'has_object', 'Item': 'has_item', 'Thing': 'has_thing',
        'Product': 'has_product', 'Goods': 'has_goods', 'Service': 'has_service',
        'Work': 'has_work', 'Artwork': 'has_artwork', 'Book': 'has_book',
        'Song': 'has_song', 'Movie': 'has_movie', 'Play': 'has_play',
        'Poem': 'has_poem', 'Novel': 'has_novel', 'Story': 'has_story',
        'Article': 'has_article', 'Paper': 'has_paper', 'Report': 'has_report',
        'Document': 'has_document', 'Text': 'has_text', 'Message': 'has_message',
        'Information': 'has_information', 'Data': 'has_data', 'Knowledge': 'has_knowledge',
        'Skill': 'has_skill', 'Ability': 'has_ability', 'Talent': 'has_talent',
        'Gift': 'has_gift', 'Present': 'has_present', 'Award': 'has_award',
        'Prize': 'has_prize', 'Medal': 'has_medal', 'Trophy': 'has_trophy',
        'Certificate': 'has_certificate', 'Degree': 'has_degree', 'Diploma': 'has_diploma',
        
        # Events and activities
        'Event': 'has_event', 'Activity': 'has_activity', 'Action': 'has_action',
        'Competition': 'has_competition', 'Game': 'has_game', 'Match': 'has_match',
        'Race': 'has_race', 'Tournament': 'has_tournament', 'Contest': 'has_contest',
        'Battle': 'has_battle', 'War': 'has_war', 'Fight': 'has_fight',
        'Conflict': 'has_conflict', 'Dispute': 'has_dispute', 'Argument': 'has_argument',
        'Meeting': 'has_meeting', 'Conference': 'has_conference', 'Summit': 'has_summit',
        'Party': 'has_party', 'Celebration': 'has_celebration', 'Festival': 'has_festival',
        'Ceremony': 'has_ceremony', 'Wedding': 'has_wedding', 'Funeral': 'has_funeral',
        'Birthday': 'has_birthday', 'Anniversary': 'has_anniversary',
        
        # Organizations and institutions
        'Organization': 'has_organization', 'Institution': 'has_institution',
        'Company': 'has_company', 'Corporation': 'has_corporation', 'Business': 'has_business',
        'School': 'has_school', 'University': 'has_university', 'College': 'has_college',
        'Hospital': 'has_hospital', 'Clinic': 'has_clinic', 'Museum': 'has_museum',
        'Library': 'has_library', 'Theater': 'has_theater', 'Cinema': 'has_cinema',
        'Stadium': 'has_stadium', 'Arena': 'has_arena', 'Auditorium': 'has_auditorium',
        'Church': 'has_church', 'Temple': 'has_temple', 'Mosque': 'has_mosque',
        'Government': 'has_government', 'State': 'has_state', 'Country': 'has_country',
        'City': 'has_city', 'Town': 'has_town', 'Village': 'has_village',
        
        # Relationships
        'Friend': 'has_friend', 'Enemy': 'has_enemy', 'Rival': 'has_rival',
        'Partner': 'has_partner', 'Spouse': 'has_spouse', 'Husband': 'has_husband',
        'Wife': 'has_wife', 'Parent': 'has_parent', 'Mother': 'has_mother',
        'Father': 'has_father', 'Child': 'has_child', 'Son': 'has_son',
        'Daughter': 'has_daughter', 'Sibling': 'has_sibling', 'Brother': 'has_brother',
        'Sister': 'has_sister', 'Family': 'has_family', 'Relative': 'has_relative',
        'Colleague': 'has_colleague', 'Boss': 'has_boss', 'Employee': 'has_employee',
        'Teacher': 'has_teacher', 'Mentor': 'has_mentor', 'Protégé': 'has_protege',
        'Leader': 'has_leader', 'Follower': 'has_follower', 'Supporter': 'has_supporter',
        'Opponent': 'has_opponent', 'Critic': 'has_critic', 'Fan': 'has_fan',
        'Admirer': 'has_admirer', 'Lover': 'has_lover',
        
        # Attributes and properties
        'Attribute': 'has_attribute', 'Property': 'has_property', 'Characteristic': 'has_characteristic',
        'Quality': 'has_quality', 'Feature': 'has_feature', 'Aspect': 'has_aspect',
        'Color': 'has_color', 'Size': 'has_size', 'Shape': 'has_shape',
        'Weight': 'has_weight', 'Height': 'has_height', 'Length': 'has_length',
        'Age': 'has_age', 'Gender': 'has_gender', 'Nationality': 'has_nationality',
        'Religion': 'has_religion', 'Language': 'has_language', 'Culture': 'has_culture',
        'Tradition': 'has_tradition', 'Custom': 'has_custom', 'Habit': 'has_habit',
        'Behavior': 'has_behavior', 'Personality': 'has_personality', 'Character': 'has_character',
        'Mood': 'has_mood', 'Emotion': 'has_emotion', 'Feeling': 'has_feeling',
        'Attitude': 'has_attitude', 'Opinion': 'has_opinion', 'Belief': 'has_belief',
        'Value': 'has_value', 'Principle': 'has_principle', 'Ideal': 'has_ideal',
        
        # Categories and types
        'Category': 'has_category', 'Type': 'has_type', 'Kind': 'has_kind',
        'Class': 'has_class', 'Group': 'has_group', 'Set': 'has_set',
        'Collection': 'has_collection', 'Series': 'has_series', 'Sequence': 'has_sequence',
        'List': 'has_list', 'Array': 'has_array', 'Batch': 'has_batch',
        'Bunch': 'has_bunch', 'Cluster': 'has_cluster', 'Crowd': 'has_crowd',
        'Team': 'has_team', 'Squad': 'has_squad', 'Crew': 'has_crew',
        'Staff': 'has_staff', 'Personnel': 'has_personnel', 'Workforce': 'has_workforce',
        
        # Results and outcomes
        'Result': 'has_result', 'Outcome': 'has_outcome', 'Consequence': 'has_consequence',
        'Effect': 'has_effect', 'Impact': 'has_impact', 'Influence': 'has_influence',
        'Change': 'has_change', 'Transformation': 'has_transformation', 'Development': 'has_development',
        'Progress': 'has_progress', 'Improvement': 'has_improvement', 'Advancement': 'has_advancement',
        'Achievement': 'has_achievement', 'Success': 'has_success', 'Victory': 'has_victory',
        'Triumph': 'has_triumph', 'Win': 'has_win', 'Loss': 'has_loss',
        'Failure': 'has_failure', 'Defeat': 'has_defeat', 'Mistake': 'has_mistake',
        'Error': 'has_error', 'Problem': 'has_problem', 'Issue': 'has_issue',
        'Challenge': 'has_challenge', 'Obstacle': 'has_obstacle', 'Difficulty': 'has_difficulty',
        
        # Methods and means
        'Method': 'has_method', 'Means': 'has_means', 'Way': 'has_way',
        'Technique': 'has_technique', 'Strategy': 'has_strategy', 'Approach': 'has_approach',
        'Tool': 'has_tool', 'Instrument': 'has_instrument', 'Device': 'has_device',
        'Machine': 'has_machine', 'Equipment': 'has_equipment', 'Apparatus': 'has_apparatus',
        'Technology': 'has_technology', 'System': 'has_system', 'Process': 'has_process',
        'Procedure': 'has_procedure', 'Protocol': 'has_protocol', 'Rule': 'has_rule',
        'Law': 'has_law', 'Regulation': 'has_regulation', 'Policy': 'has_policy',
        'Guideline': 'has_guideline', 'Standard': 'has_standard', 'Criterion': 'has_criterion',
        
        # Reasons and purposes
        'Reason': 'has_reason', 'Cause': 'has_cause', 'Purpose': 'has_purpose',
        'Goal': 'has_goal', 'Objective': 'has_objective', 'Target': 'has_target',
        'Aim': 'has_aim', 'Intention': 'has_intention', 'Plan': 'has_plan',
        'Scheme': 'has_scheme', 'Project': 'has_project', 'Mission': 'has_mission',
        'Task': 'has_task', 'Job': 'has_job', 'Work': 'has_work',
        'Labor': 'has_labor', 'Effort': 'has_effort', 'Activity': 'has_activity',
        'Action': 'has_action', 'Deed': 'has_deed', 'Act': 'has_act',
        'Operation': 'has_operation', 'Function': 'has_function', 'Role': 'has_role',
        'Position': 'has_position', 'Status': 'has_status', 'Rank': 'has_rank',
        'Level': 'has_level', 'Degree': 'has_degree', 'Grade': 'has_grade',
        'Score': 'has_score', 'Rating': 'has_rating', 'Evaluation': 'has_evaluation',
        'Assessment': 'has_assessment', 'Judgment': 'has_judgment', 'Opinion': 'has_opinion',
        'View': 'has_view', 'Perspective': 'has_perspective', 'Standpoint': 'has_standpoint',
        'Stance': 'has_stance'
    }
    
    return frame_specific, generic_mapping

def get_predicate_for_role(role: str, frame_name: str) -> str:
    """Get predicate name for a semantic role with comprehensive frame support."""
    frame_specific, generic_mapping = get_comprehensive_frame_mappings()
    
    # Check frame-specific mapping first
    if frame_name in frame_specific and role in frame_specific[frame_name]:
        return frame_specific[frame_name][role]
    
    # Fall back to generic mapping
    return generic_mapping.get(role, 'has_theme')

if __name__ == "__main__":
    # Test the comprehensive mappings
    frame_specific, generic_mapping = get_comprehensive_frame_mappings()
    
    print(f"Frame-specific mappings: {len(frame_specific)} frames")
    print(f"Generic mappings: {len(generic_mapping)} roles")
    
    # Test some examples
    test_cases = [
        ('Being_born', 'Child'),
        ('Win_prize', 'Competitor'),
        ('Leadership', 'Leader'),
        ('Motion', 'Theme'),
        ('Unknown_frame', 'Agent'),
        ('Unknown_frame', 'Unknown_role')
    ]
    
    for frame, role in test_cases:
        predicate = get_predicate_for_role(role, frame)
        print(f"{frame}.{role} -> {predicate}")
