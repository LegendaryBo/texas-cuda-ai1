package pl.wroc.uni.ii.evolution.experimental;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;

import pl.wroc.uni.ii.evolution.engine.prototype.EvObjectiveFunction;
import pl.wroc.uni.ii.evolution.engine.prototype.EvSolutionSpace;
import pl.wroc.uni.ii.evolution.experimental.decisiontree.EvAnswer;
import pl.wroc.uni.ii.evolution.experimental.decisiontree.EvDecision;
import pl.wroc.uni.ii.evolution.experimental.decisiontree.EvDecisionTreeIndividual;

/**
 * TODO SolutionSpace for decision trees.
 * 
 * @author Marcin Golebiowski
 */
public class EvDecisionTreesSpace<T> implements
    EvSolutionSpace<EvDecisionTreeIndividual<T>> {
  private static final long serialVersionUID = -2619008445373412303L;

  private Map<String, EvDecision<T>> possible_nodes;


  /**
   * Basic constructor
   */
  public EvDecisionTreesSpace() {
    possible_nodes = new HashMap<String, EvDecision<T>>();
  }


  /**
   * @return
   */
  public ArrayList<EvDecision<T>> returnPossibleNodes() {
    ArrayList<EvDecision<T>> nodes = new ArrayList<EvDecision<T>>();
    nodes.addAll(possible_nodes.values());
    return nodes;
  }


  /**
   * @param decider
   */
  public void addPossibleNode(EvDecision<T> decider) {
    possible_nodes.put(decider.getClass().getName(), decider);
  }


  public boolean belongsTo(EvDecisionTreeIndividual<T> individual) {
    EvDecision<T> decider = individual.getDecision_strategy();
    if (!possible_nodes.containsKey(decider.getClass().getName())) {
      return false;
    }

    Map<EvAnswer, EvDecisionTreeIndividual<T>> children =
        individual.getChildren();

    if (children.containsKey(EvAnswer.YES)) {
      if (!belongsTo(children.get(EvAnswer.YES))) {
        return false;
      }
    }

    if (children.containsKey(EvAnswer.NO)) {
      if (!belongsTo(children.get(EvAnswer.NO))) {
        return false;
      }
    }

    if (children.containsKey(EvAnswer.DUNNO)) {
      if (!belongsTo(children.get(EvAnswer.DUNNO))) {
        return false;
      }
    }

    return true;
  }


  public Set<EvSolutionSpace<EvDecisionTreeIndividual<T>>> divide(int n) {
    return null;
  }


  public Set<EvSolutionSpace<EvDecisionTreeIndividual<T>>> divide(int n,
      Set<EvDecisionTreeIndividual<T>> p) {
    return null;
  }


  // TODO W jaki sposób wygenerowac losowe drzewo ?
  // jak ograniczyc glêbokoœæ drzewa
  //

  public EvDecisionTreeIndividual<T> generateIndividual() {
    return null;
  }


  public EvDecisionTreeIndividual<T> takeBackTo(
      EvDecisionTreeIndividual<T> individual) {
    return null;
  }


  /* is there any way to avoid implementing this function? */
  /**
   * this method has no use
   */
  public void setObjectiveFuntion(
      EvObjectiveFunction<EvDecisionTreeIndividual<T>> objective_function) {
  }


  /**
   * this method return null
   */
  public EvObjectiveFunction<EvDecisionTreeIndividual<T>> getObjectiveFuntion() {
    return null;
  }

}