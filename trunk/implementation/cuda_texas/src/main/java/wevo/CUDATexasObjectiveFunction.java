package wevo;

import engine.TexasSettings;
import generator.GeneratorGraczyZGeneracji;
import generator.ProstyGeneratorLiczb;
import pl.wroc.uni.ii.evolution.engine.individuals.EvBinaryVectorIndividual;
import pl.wroc.uni.ii.evolution.engine.prototype.EvObjectiveFunction;
import Gracze.gracz_v3.GeneratorRegulv3;
import cuda.swig.SWIGTYPE_p_int;
import cuda.swig.SWIGTYPE_p_p_int;
import cuda.swig.ai_texas_swig;

//TODO trzeba zrobic tak, zeby funkcja celu na CPU i GPU zwracaly taki sam wynik
//TODO optymalizacje

/**
 * Funkcja celu, ktora oblicza osobnika na karcie graficznej.
 * 
 * @author Kacper Gorski (railman85@gmail.com)
 */
public class CUDATexasObjectiveFunction implements EvObjectiveFunction<EvBinaryVectorIndividual> {

	private static final long serialVersionUID = 8765187926167966922L;
	private final int LICZBA_OSOBNIKOW = 100; // wiecej sie nie miesci na karte :(
	private final int LICZBA_GENOW = GeneratorRegulv3.rozmiarGenomu;
	private final int LICZBA_INTOW = (GeneratorRegulv3.rozmiarGenomu - 1) / 32 + 1;
	private final ProstyGeneratorLiczb random = new ProstyGeneratorLiczb(465);

	private final int liczbaWatkowNaBlok;
	private final int liczbaPartii;
	private final SWIGTYPE_p_p_int osobniki_ptr;
	private final GeneratorGraczyZGeneracji generator;
	private SWIGTYPE_p_int[] osobnikiTreningowe = null;

	/**
	 * 
	 * @param liczbaGeneracji
	 *                (ile generacji ma grac z osobnikiem, dac na 11)
	 * @param liczbaWatkowNaBlok
	 *                - liczba watkow uzytych do rozegrania jednej partii
	 * @param liczbaPartii
	 *                - liczba partii
	 */
	public CUDATexasObjectiveFunction(int liczbaGeneracji, int liczbaWatkowNaBlok, int liczbaPartii) {
		TexasSettings.setTexasLibraryPath();
		this.liczbaPartii = liczbaPartii;
		this.liczbaWatkowNaBlok = liczbaWatkowNaBlok;
		generator = new GeneratorGraczyZGeneracji(1234, LICZBA_GENOW, liczbaGeneracji, true);
		osobniki_ptr = ai_texas_swig.getIndividualPTRPTR(LICZBA_OSOBNIKOW + 1);
		init();
	}

	// funkcja kopiuje do pamieci RAM osobniki treningowe
	private void init() {
		EvBinaryVectorIndividual[] osobniki_java = new EvBinaryVectorIndividual[LICZBA_OSOBNIKOW];
		int[][] osobniki = new int[LICZBA_OSOBNIKOW][];
		for (int i = 0; i < LICZBA_OSOBNIKOW; i++) {
			int losowaLiczba = random.nextInt(generator.lista.size());
			osobniki_java[i] = generator.lista.get(losowaLiczba);
			osobniki[i] = osobniki_java[i].getGenes();
		}

		osobnikiTreningowe = new SWIGTYPE_p_int[osobniki.length];
		for (int i = 0; i < osobniki.length; i++) {
			SWIGTYPE_p_int osobnik_ptr = ai_texas_swig.getOsobnikPTR(osobniki[i], LICZBA_INTOW);
			ai_texas_swig.setIndividualPTR(osobnik_ptr, osobniki_ptr, i);
			osobnikiTreningowe[i] = osobnik_ptr;
		}
	}

	public double evaluate(EvBinaryVectorIndividual obliczany_osobnik) {

		SWIGTYPE_p_int wskaznikDoPamieci = kopiujOsobnikaDoPamieci(obliczany_osobnik);

		float[] wynik_cuda = new float[1]; // bo w swigu tak pokracznie sie zwraca wyniki :)
		ai_texas_swig.rozegrajNGierCUDA(LICZBA_OSOBNIKOW, osobniki_ptr, wynik_cuda, liczbaPartii, LICZBA_INTOW, liczbaWatkowNaBlok);

		// usuwanie obliczonego osobnika z pamieci ram (jednak treningowe osobniki zostaja na zawsze!)
		ai_texas_swig.destruktorInt(wskaznikDoPamieci);

		return wynik_cuda[0];
	}

	private SWIGTYPE_p_int kopiujOsobnikaDoPamieci(EvBinaryVectorIndividual obliczany_osobnik) {
		SWIGTYPE_p_int obliczany_osobnik_ptr = ai_texas_swig.getOsobnikPTR(obliczany_osobnik.getGenes(), LICZBA_INTOW);
		ai_texas_swig.setIndividualPTR(obliczany_osobnik_ptr, osobniki_ptr, LICZBA_OSOBNIKOW);
		return obliczany_osobnik_ptr;
	}

	public void usunOsobnikiTreningoweZPamieci() {
		for (int i = 0; i < LICZBA_OSOBNIKOW; i++) {
			ai_texas_swig.destruktorInt(osobnikiTreningowe[i]);
		}
	}

	@Override
	public void finalize() {
//		usunOsobnikiTreningoweZPamieci();
	}
}
