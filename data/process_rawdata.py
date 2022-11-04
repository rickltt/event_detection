import json
import os
import argparse
import random
# maven_labels = ['Know', 'Warning', 'Catastrophe', 'Placing', 'Causation', 'Arriving', 'Sending', 'Protest', 'Preventing_or_letting', 'Motion', 'Damaging', 'Destroying', 'Death', 'Perception_active', 'Presence', 'Influence', 'Receiving', 'Check', 'Hostile_encounter', 'Killing', 'Conquering', 'Releasing', 'Attack', 'Earnings_and_losses', 'Choosing', 'Traveling', 'Recovering', 'Using', 'Coming_to_be', 'Cause_to_be_included', 'Process_start', 'Change_event_time', 'Reporting', 'Bodily_harm', 'Suspicion', 'Statement', 'Cause_change_of_position_on_a_scale', 'Coming_to_believe', 'Expressing_publicly', 'Request', 'Control', 'Supporting', 'Defending', 'Building', 'Military_operation', 'Self_motion', 'GetReady', 'Forming_relationships', 'Becoming_a_member', 'Action', 'Removing', 'Surrendering', 'Agree_or_refuse_to_act', 'Participation', 'Deciding', 'Education_teaching', 'Emptying', 'Getting', 'Besieging', 'Creating', 'Process_end', 'Body_movement', 'Expansion', 'Telling', 'Change', 'Legal_rulings', 'Bearing_arms', 'Giving', 'Name_conferral', 'Arranging', 'Use_firearm', 'Committing_crime', 'Assistance', 'Surrounding', 'Quarreling', 'Expend_resource', 'Motion_directional', 'Bringing', 'Communication', 'Containing', 'Manufacturing', 'Social_event', 'Robbery', 'Competition', 'Writing', 'Rescuing', 'Judgment_communication', 'Change_tool', 'Hold', 'Being_in_operation', 'Recording', 'Carry_goods', 'Cost', 'Departing', 'GiveUp', 'Change_of_leadership', 'Escaping', 'Aiming', 'Hindering', 'Preserving', 'Create_artwork', 'Openness', 'Connect', 'Reveal_secret', 'Response', 'Scrutiny', 'Lighting', 'Criminal_investigation', 'Hiding_objects', 'Confronting_problem', 'Renting', 'Breathing', 'Patrolling', 'Arrest', 'Convincing', 'Commerce_sell', 'Cure', 'Temporary_stay', 'Dispersal', 'Collaboration', 'Extradition', 'Change_sentiment', 'Commitment', 'Commerce_pay', 'Filling', 'Becoming', 'Achieve', 'Practice', 'Cause_change_of_strength', 'Supply', 'Cause_to_amalgamate', 'Scouring', 'Violence', 'Reforming_a_system', 'Come_together', 'Wearing', 'Cause_to_make_progress', 'Legality', 'Employment', 'Rite', 'Publishing', 'Adducing', 'Exchange', 'Ratification', 'Sign_agreement', 'Commerce_buy', 'Imposing_obligation', 'Rewards_and_punishments', 'Institutionalization', 'Testing', 'Ingestion', 'Labeling', 'Kidnapping', 'Submitting_documents', 'Prison', 'Justifying', 'Emergency', 'Terrorism', 'Vocalizations', 'Risk', 'Resolve_problem', 'Revenge', 'Limiting', 'Research', 'Having_or_lacking_access', 'Theft', 'Incident', 'Award']
# print(len(maven_labels))

ace_labels = ["Attack", "Transport", "Die", "End-Position", "Meet", "Phone-Write", "Elect", "Injure", "Transfer-Ownership", "Start-Org", "Transfer-Money", "Sue", "Demonstrate", "Arrest-Jail", "Start-Position", "Be-Born", "End-Org", "Execute", "Nominate", "Fine", "Trial-Hearing", "Marry", "Charge-Indict", "Sentence",  "Convict", "Appeal", "Declare-Bankruptcy", "Merge-Org", "Release-Parole", "Pardon", "Extradite", "Divorce", "Acquit"]
# print(len(ace_labels))

cnc_labels = ['Geopolitical-tension', 'Movement-down-loss', 'Position-low', 'trade-financial-tension', 'Movement-up-gain', 'Slow-weak', 'Negative_sentiment', 'Grow-strong', 'Civil-unrest', 'Embargo', 'Oversupply', 'Cause-movement-down-loss', 'Position-high', 'Prohibiting', 'Situation_deteriorate', 'Shortage', 'Movement-flat', 'Crisis', 'Cause-movement-up-gain']
# print(len(cnc_labels))


def get_cnc(args):
    data_dir = args.data_dir_input + 'cnc'
    output_dir = args.data_dir_output + 'cnc'

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)   
    datafile_input = ['train','test']   
    data = {}
    data['train'] = []
    data['test'] = []
    data['dev'] = []

    for mode in datafile_input:
        with open(os.path.join(data_dir,'event_extraction_'+mode+'.json'),"r") as f:
           instances = json.load(f)
           for instance in instances:
                sample = {}
                sample['sentence'] = instance['sentence']
                sample['tokens'] = instance['words']
                sample['events'] = []
                for event in instance["golden-event-mentions"]:
                    trigger = event["trigger"]["text"]
                    event_type = event["event_type"]
                    # if event_type not in event_list:
                    #     event_list.append(event_type)
                    start = event["trigger"]["start"]
                    end = event["trigger"]["end"]
                    sample['events'].append({
                        "trigger":trigger,
                        "event_type":event_type,
                        "start":start,
                        "end":end
                    })
                data[mode].append(sample)
    
    len_dev = len(data['test'])
    random.shuffle(data['train'])
    data['dev'] = data['train'][:len_dev]
    data['train'] = data['train'][len_dev:]

    for mode in data:
        for i,j in enumerate(data[mode]):
            j['guid'] = str(mode) + '-' + str(i+1)

    for mode in data:
        with open(os.path.join(output_dir,mode+'.json'),"w") as fw:
            json.dump(data[mode],fw,indent=2,ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, default="cnc")
    parser.add_argument("--data_dir_input", type=str, default="./raw_data/")
    parser.add_argument("--data_dir_output", type=str, default="./")

    args = parser.parse_args()
    
    if args.dataset == 'cnc':
        get_cnc(args)

    elif args.dataset == 'all':
        get_cnc(args)

if __name__ == '__main__':
    main()