package pl.wroc.uni.ii.evolution.distribution.strategies;

import java.util.List;
import java.util.Set;
import java.util.TreeSet;

import org.jmock.Mock;
import org.jmock.MockObjectTestCase;

import pl.wroc.uni.ii.evolution.distribution.strategies.EvIslandModel;
import pl.wroc.uni.ii.evolution.distribution.strategies.exchange.EvReceiver;
import pl.wroc.uni.ii.evolution.distribution.strategies.exchange.EvSender;
import pl.wroc.uni.ii.evolution.distribution.workers.EvBlankEvolInterface;
import pl.wroc.uni.ii.evolution.engine.EvPopulation;
import pl.wroc.uni.ii.evolution.engine.operators.general.replacement.EvBestFromUnionReplacement;
import pl.wroc.uni.ii.evolution.engine.operators.general.selections.EvKBestSelection;
import pl.wroc.uni.ii.evolution.engine.prototype.EvOperator;
import pl.wroc.uni.ii.evolution.engine.prototype.EvReplacement;
import pl.wroc.uni.ii.evolution.testhelper.EvStunt;

public class EvDistributionStrategyTest extends MockObjectTestCase {
  private List<EvStunt> four_individuals;
  
  EvIslandModel<EvStunt> distr;
  Mock exporter,importer;
  
  final long signature = 1L;
  @SuppressWarnings("unchecked")
  @Override
  protected void setUp() throws Exception {
    exporter = mock(EvSender.class);
    importer = mock(EvReceiver.class);
    EvReplacement<EvStunt> replacement = new EvBestFromUnionReplacement<EvStunt>();
    EvOperator<EvStunt> export_selection = 
      new EvKBestSelection<EvStunt>(2);
    
    distr = 
      new EvIslandModel<EvStunt>(
          (EvReceiver)importer.proxy(), (EvSender) exporter.proxy(), 
          replacement, export_selection );

    four_individuals = EvStunt.list(1,2,3,4);
  }

  public void testStartingThreads() throws Exception {
    importer.expects(once()).method("init");
    exporter.expects(once()).method("init");
    importer.expects(once()).method("start");
    exporter.expects(once()).method("start");
    distr.init(new EvBlankEvolInterface());
  }
  
  public void testSelectionBeforeExport() throws Exception {
    importer.expects(once()).method("init");
    exporter.expects(once()).method("init");
    importer.expects(once()).method("start");
    exporter.expects(once()).method("start");
    distr.init(new EvBlankEvolInterface());
    
    exporter.expects(once()).method("export").with(eq(EvStunt.list(4,3)));
    distr.export(population(four_individuals));
  }
  
  public void testReplacementAfterImport() throws Exception {
    importer.expects(once()).method("init");
    exporter.expects(once()).method("init");
    importer.expects(once()).method("start");
    exporter.expects(once()).method("start");
    distr.init(new EvBlankEvolInterface());
    
    importer.expects(once()).method("getIndividuals").will(returnValue(EvStunt.list(5,6)));
    EvPopulation<EvStunt> population = population(four_individuals);
    distr.updatePopulation(population);
    
    assertEquals(EvStunt.set(3,4,5,6)
        , set(population));
  }
  
  private EvPopulation<EvStunt> population(List<EvStunt> four_individuals2) {
    EvPopulation<EvStunt> population = new EvPopulation<EvStunt>();
    for(EvStunt indiv : four_individuals2)
      population.add(indiv);
    return population;
  }

  private Set<EvStunt> set(List<EvStunt> list) {
    return new TreeSet<EvStunt>(list);
  }
  
}
