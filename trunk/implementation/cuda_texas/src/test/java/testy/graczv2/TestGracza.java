package testy.graczv2;

import engine.Gra;
import generator.IndividualGenerator;
import junit.framework.TestCase;
import Gracze.GraczAIv2;
import Gracze.gracz_v2.generatorRegul.GeneratorRegul;

/**
 * 
 * @author Kacper Gorski
 *
 */
public class TestGracza extends TestCase {

  public void testProsty() {
    
    GeneratorRegul.init();
    final int pGenes = GeneratorRegul.rozmiarGenomu;
    
    GraczAIv2[] pGracze = new GraczAIv2[6];
    
    IndividualGenerator pGenerator = new IndividualGenerator(13, pGenes);
    
    pGracze[0] = new GraczAIv2(pGenerator.generate(),0);
    pGracze[1] = new GraczAIv2(pGenerator.generate(),1);
    pGracze[2] = new GraczAIv2(pGenerator.generate(),2);
    pGracze[3] = new GraczAIv2(pGenerator.generate(),3);
    pGracze[4] = new GraczAIv2(pGenerator.generate(),4);
    pGracze[5] = new GraczAIv2(pGenerator.generate(),5);
    
    Gra pGra = new Gra(pGracze ,13);
    
    pGra.play_round(false);
    
  }
  
}
