from typing import Union, List
import nenv
from nenv import Session, SessionEstimator, Bid
from nenv.utils import LogRow, ExcelLog
from .NegoFormerAgent import NegoFormerAgent


class CandidatesLogger(nenv.logger.AbstractLogger):
    """
        This logger analyzes the candidate selection for Negoformer Agent
        It generates selection distribution.
    """
    agent_pos: str = ""
    is_estimator_session: bool
    candidates_list: List[str] = ["Pareto", "Nash", "Kalai", "MaxOpp", "Center"]

    def before_session_start(self, session: Union[Session, SessionEstimator]) -> List[str]:
        """
            This method checks whether NegoFormerAgent will negotiate in this session.
        """
        if isinstance(session, SessionEstimator):
            self.is_estimator_session = True
        else:
            self.is_estimator_session = False

        if session.agentA.name == "NegoFormerAgent":
            self.agent_pos = "A"

            return ["NegoFormerAgent_Candidates"]
        elif session.agentB.name == "NegoFormerAgent":
            self.agent_pos = "B"

            return ["NegoFormerAgent_Candidates"]
        else:
            self.agent_pos = ""

            return []

    def on_offer(self, agent: str, offer: Bid, time: float, session: Union[Session, SessionEstimator]) -> LogRow:
        if self.agent_pos == "" or self.is_estimator_session:
            return {}

        if self.agent_pos != agent:
            return {"NegoFormerAgent_Candidates": {"Selected": "-",
                                                     "Pareto_Slope": "-", "Pareto_AgentUtility": "-", "Pareto_OppUtility": "-", "Pareto_EstOppUtility": "-",
                                                     "Nash_Slope": "-", "Nash_AgentUtility": "-", "Nash_OppUtility": "-", "Nash_EstOppUtility": "-",
                                                     "Kalai_Slope": "-", "Kalai_AgentUtility": "-", "Kalai_OppUtility": "-", "Kalai_EstOppUtility": "-",
                                                     "MaxOpp_Slope": "-", "MaxOpp_AgentUtility": "-", "MaxOpp_OppUtility": "-", "MaxOpp_EstOppUtility": "-",
                                                     "Center_Slope": "-", "Center_AgentUtility": "-", "Center_OppUtility": "-", "Center_EstOppUtility": "-", }}

        agent: NegoFormerAgent = session.agentA if self.agent_pos == "A" else session.agentB

        if len(agent.last_candidate) > 0:
            last_candidate = agent.last_candidate

            row = {}

            selected_key = min([key for key in last_candidate], key=lambda k: (last_candidate[k][0], -last_candidate[k][1].utility_a))

            row["Selected"] = selected_key

            for key in last_candidate:
                row[f'{key}_Slope'] = last_candidate[key][0]

                row[f'{key}_AgentUtility'] = agent.preference.get_utility(last_candidate[key][1].bid)

                if self.agent_pos == 'A':
                    row[f'{key}_OppUtility'] = session.agentB.preference.get_utility(last_candidate[key][1].bid)
                else:
                    row[f'{key}_OppUtility'] = session.agentA.preference.get_utility(last_candidate[key][1].bid)

                row[f'{key}_EstOppUtility'] = agent.opponent_model.preference.get_utility(last_candidate[key][1].bid)

            return {"NegoFormerAgent_Candidates": row}

        return {"NegoFormerAgent_Candidates": {"Selected": "-",
                                                 "Pareto_Slope": "-", "Pareto_AgentUtility": "-", "Pareto_OppUtility": "-", "Pareto_EstOppUtility": "-",
                                                 "Nash_Slope": "-", "Nash_AgentUtility": "-", "Nash_OppUtility": "-", "Nash_EstOppUtility": "-",
                                                 "Kalai_Slope": "-", "Kalai_AgentUtility": "-", "Kalai_OppUtility": "-", "Kalai_EstOppUtility": "-",
                                                 "MaxOpp_Slope": "-", "MaxOpp_AgentUtility": "-", "MaxOpp_OppUtility": "-", "MaxOpp_EstOppUtility": "-",
                                                 "Center_Slope": "-", "Center_AgentUtility": "-", "Center_OppUtility": "-", "Center_EstOppUtility": "-"}}

    def on_session_end(self, final_row: LogRow, session: Union[Session, SessionEstimator]) -> LogRow:
        if self.agent_pos == '':
            return {"NegoFormerAgent_Candidates": {"Pareto_Slope": "-", "Pareto_AgentUtility": "-", "Pareto_OppUtility": "-", "Pareto_EstOppUtility": "-", "Pareto_SelectionRate": "-",
                                                     "Nash_Slope": "-", "Nash_AgentUtility": "-", "Nash_OppUtility": "-", "Nash_EstOppUtility": "-", "Nash_SelectionRate": "-",
                                                     "Kalai_Slope": "-", "Kalai_AgentUtility": "-", "Kalai_OppUtility": "-", "Kalai_EstOppUtility": "-", "Kalai_SelectionRate": "-",
                                                     "MaxOpp_Slope": "-", "MaxOpp_AgentUtility": "-", "MaxOpp_OppUtility": "-", "MaxOpp_EstOppUtility": "-", "MaxOpp_SelectionRate": "-",
                                                     "Center_Slope": "-", "Center_AgentUtility": "-", "Center_OppUtility": "-", "Center_EstOppUtility": "-", "Center_SelectionRate": "-"}}

        in_session_slopes = {candidate_name: 0. for candidate_name in self.candidates_list}
        in_session_agent_utilities = {candidate_name: 0. for candidate_name in self.candidates_list}
        in_session_opp_utilities = {candidate_name: 0. for candidate_name in self.candidates_list}
        in_session_est_opp_utilities = {candidate_name: 0. for candidate_name in self.candidates_list}
        in_session_selections = {candidate_name: 0. for candidate_name in self.candidates_list}
        counter = 0

        for row in session.session_log.log_rows["NegoFormerAgent_Candidates"]:
            if "Selected" not in row or row["Selected"] == "-":
                continue

            for candidate_name in self.candidates_list:
                in_session_slopes[candidate_name] += float(row[f'{candidate_name}_Slope'])
                in_session_agent_utilities[candidate_name] += float(row[f'{candidate_name}_AgentUtility'])
                in_session_opp_utilities[candidate_name] += float(row[f'{candidate_name}_OppUtility'])
                in_session_est_opp_utilities[candidate_name] += float(row[f'{candidate_name}_EstOppUtility'])

                if row["Selected"] == candidate_name:
                    in_session_selections[candidate_name] += 1

            counter += 1

        if counter == 0:
            return {"NegoFormerAgent_Candidates": {"Pareto_Slope": "-", "Pareto_AgentUtility": "-", "Pareto_OppUtility": "-", "Pareto_EstOppUtility": "-", "Pareto_SelectionRate": "-",
                                                     "Nash_Slope": "-", "Nash_AgentUtility": "-", "Nash_OppUtility": "-", "Nash_EstOppUtility": "-", "Nash_SelectionRate": "-",
                                                     "Kalai_Slope": "-", "Kalai_AgentUtility": "-", "Kalai_OppUtility": "-", "Kalai_EstOppUtility": "-", "Kalai_SelectionRate": "-",
                                                     "MaxOpp_Slope": "-", "MaxOpp_AgentUtility": "-", "MaxOpp_OppUtility": "-", "MaxOpp_EstOppUtility": "-", "MaxOpp_SelectionRate": "-",
                                                     "Center_Slope": "-", "Center_AgentUtility": "-", "Center_OppUtility": "-", "Center_EstOppUtility": "-", "Center_SelectionRate": "-"}}


        in_session_slopes = {key: in_session_slopes[key] / counter for key in self.candidates_list}
        in_session_agent_utilities = {key: in_session_agent_utilities[key] / counter for key in self.candidates_list}
        in_session_opp_utilities = {key: in_session_opp_utilities[key] / counter for key in self.candidates_list}
        in_session_est_opp_utilities = {key: in_session_est_opp_utilities[key] / counter for key in self.candidates_list}

        in_session_selections = {key: in_session_selections[key] / sum(in_session_selections.values()) for key in self.candidates_list}

        row = {}

        for key in self.candidates_list:
            row[f'{key}_Slope'] = in_session_slopes[key]

            row[f'{key}_AgentUtility'] = in_session_agent_utilities[key]

            row[f'{key}_OppUtility'] = in_session_opp_utilities[key]

            row[f'{key}_EstOppUtility'] = in_session_est_opp_utilities[key]

            row[f'{key}_SelectionRate'] = in_session_selections[key]

        return {"NegoFormerAgent_Candidates": row}

    def on_tournament_end(self, tournament_logs: ExcelLog, agent_names: List[str], domain_names: List[str], estimator_names: List[str]):
        overall_selections = {candidate_name: 0. for candidate_name in self.candidates_list}
        overall_slopes = {candidate_name: 0. for candidate_name in self.candidates_list}
        overall_agent_utilities = {candidate_name: 0. for candidate_name in self.candidates_list}
        overall_opp_utilities = {candidate_name: 0. for candidate_name in self.candidates_list}
        overall_est_opp_utilities = {candidate_name: 0. for candidate_name in self.candidates_list}
        counter = 0

        for row in tournament_logs.log_rows["NegoFormerAgent_Candidates"]:
            if "Pareto_Slope" not in row or row["Pareto_Slope"] == "-":
                continue

            for candidate_name in self.candidates_list:
                overall_slopes[candidate_name] += float(row[f'{candidate_name}_Slope'])
                overall_agent_utilities[candidate_name] += float(row[f'{candidate_name}_AgentUtility'])
                overall_opp_utilities[candidate_name] += float(row[f'{candidate_name}_OppUtility'])
                overall_est_opp_utilities[candidate_name] += float(row[f'{candidate_name}_EstOppUtility'])
                overall_selections[candidate_name] += float(row[f'{candidate_name}_SelectionRate'])

            counter += 1

        if counter == 0:
            return

        with open(self.get_path("CandidatesLogs.csv"), "w") as f:
            for key in self.candidates_list:
                f.write(f'{key}_Slope;')
                f.write(f'{key}_AgentUtility;')
                f.write(f'{key}_OppUtility;')
                f.write(f'{key}_EstOppUtility;')
                f.write(f'{key}_SelectionRate;')

            f.write("\n")
            for key in self.candidates_list:
                f.write(str(overall_slopes[key] / counter) + ";")
                f.write(str(overall_agent_utilities[key] / counter) + ";")
                f.write(str(overall_opp_utilities[key] / counter) + ";")
                f.write(str(overall_est_opp_utilities[key] / counter) + ";")
                f.write(str(overall_selections[key] / sum(overall_selections.values())) + ";")

            f.write("\n")
