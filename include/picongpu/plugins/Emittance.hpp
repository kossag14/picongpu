/* Copyright 2013-2018 Axel Huebl, Felix Schmitt, Heiko Burau,
 *                     Rene Widera, Richard Pausch, Benjamin Worpitz
 *
 * This file is part of PIConGPU.
 *
 * PIConGPU is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * PIConGPU is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with PIConGPU.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "picongpu/simulation_defines.hpp"
#include "picongpu/algorithms/KinEnergy.hpp"
#include "picongpu/plugins/common/txtFileHandling.hpp"
#include "picongpu/plugins/multi/multi.hpp"
#include "picongpu/particles/traits/SpeciesEligibleForSolver.hpp"
#include "picongpu/particles/traits/GenerateSolversIfSpeciesEligible.hpp"
#include "picongpu/plugins/misc/misc.hpp"
#include "picongpu/simulationControl/MovingWindow.hpp"

#include <pmacc/mappings/kernel/AreaMapping.hpp>
#include <pmacc/mpi/reduceMethods/Reduce.hpp>
#include <pmacc/mpi/MPIReduce.hpp>
#include <pmacc/nvidia/functors/Add.hpp>
#include <pmacc/memory/shared/Allocate.hpp>
#include <pmacc/dataManagement/DataConnector.hpp>
#include <pmacc/nvidia/atomic.hpp>
#include <pmacc/mappings/threads/ForEachIdx.hpp>
#include <pmacc/mappings/threads/IdxConfig.hpp>
#include <pmacc/memory/CtxArray.hpp>
#include <pmacc/traits/GetNumWorkers.hpp>
#include <pmacc/traits/HasIdentifiers.hpp>
#include <pmacc/traits/HasFlag.hpp>
#include <pmacc/algorithms/ForEach.hpp>
#include <pmacc/math/Vector.hpp>

#include <boost/mpl/and.hpp>

#include <string>
#include <iostream>
#include <fstream>
#include <memory>
#include <stdexcept>




namespace picongpu
{

    /** calculates the emittance in x direction
     */
    template< uint32_t T_numWorkers >
    struct KernelCalcEmittance
    {

        /** calculates the sum of x², ux² and (x*ux)² and counts electrons
         *
         * @tparam T_ParBox pmacc::ParticlesBox, particle box type
         * @tparam T_DBox pmacc::DataBox, type of the memory box for the reduced values
         * @tparam T_Mapping mapper functor type
         *
         * @param pb particle memory
         * @param gSum storage for the reduced values
         *                (two elements 0 == sum of x²; 1 == sum of ux²; 2 == sum of (x*ux)²; 3 == counts electrons)
         * @param mapper functor to map a block to a supercell
         */
        template<
            typename T_ParBox,
            typename T_DBox,
            typename T_Mapping,
            typename T_Acc,
            typename T_Filter
        >
        DINLINE void operator( )(
            T_Acc const & acc,
            T_ParBox pb,
            T_DBox gSumMom2,
            T_DBox gSumPos2,
            T_DBox gSumMomPos2,
            T_DBox gCount_e,
            DataSpace<simDim> globalOffset,
            const int subGridY,
            T_Mapping mapper,
            T_Filter filter
        ) const
        {
            using namespace mappings::threads;
            constexpr uint32_t numWorkers = T_numWorkers;
            constexpr uint32_t numParticlesPerFrame = pmacc::math::CT::volume<
                typename T_ParBox::FrameType::SuperCellSize
            >::type::value;

            uint32_t const workerIdx = threadIdx.x;

            using FramePtr = typename T_ParBox::FramePtr;

            // shared sums of x², ux², (x*ux)², particle counter 
            PMACC_SMEM( acc, shSumMom2, memory::Array< float_X,SuperCellSize::y::value > );
            PMACC_SMEM( acc, shSumPos2, memory::Array< float_X,SuperCellSize::y::value > );
            PMACC_SMEM( acc, shSumMomPos2, memory::Array< float_X,SuperCellSize::y::value > );
            PMACC_SMEM( acc, shCount_e, memory::Array< float_X,SuperCellSize::y::value > );

            using ParticleDomCfg = IdxConfig<
                numParticlesPerFrame,
                numWorkers
            >;

            using SuperCellYDom = IdxConfig<
                SuperCellSize::y::value,
                numWorkers
            >;


            ForEachIdx< SuperCellYDom >{ workerIdx }(
                [&](
                    uint32_t const linearIdx,
                    uint32_t const
                )
                {
                    // set shared sums of x², ux², (x*ux)² to zero
                    shSumMom2[linearIdx] = float_X(0.0);
					shSumPos2[linearIdx] = float_X(0.0);
					shSumMomPos2[linearIdx] = float_X(0.0);
					shCount_e[linearIdx] = float_X(0.0);
                }
            );


            __syncthreads( );

            DataSpace< simDim > const superCellIdx( mapper.getSuperCellIndex(
                DataSpace< simDim >( blockIdx )
            ));

            // each virtual thread is working on an own frame
            FramePtr frame = pb.getLastFrame( superCellIdx );

            // end kernel if we have no frames within the supercell
            if( !frame.isValid( ) )
                return;

            auto accFilter = filter(
                acc,
                superCellIdx - mapper.getGuardingSuperCells( ),
                WorkerCfg< numWorkers >{ workerIdx }
            );

            memory::CtxArray<
                typename FramePtr::type::ParticleType,
                ParticleDomCfg
            >
            currentParticleCtx(
                workerIdx,
                [&](
                    uint32_t const linearIdx,
                    uint32_t const
                )
                {
                    auto particle = frame[ linearIdx ];
                    /* - only particles from the last frame must be checked
                     * - all other particles are always valid
                     */
                    if( particle[ multiMask_ ] != 1 )
                        particle.setHandleInvalid( );
                    return particle;
                }
            );

            while( frame.isValid( ) )
            {
                // loop over all particles in the frame
                ForEachIdx< ParticleDomCfg > forEachParticle( workerIdx );

			//	int num_intervals = 10; 
			//	std::vector<float> counts(10);

                forEachParticle(
                    [&](
                        uint32_t const linearIdx,
                        uint32_t const idx
                    )
                    {
                        /* get one particle */
                        auto & particle = currentParticleCtx[ idx ];
                        if(accFilter(acc,particle)){
							
                            float_X const weighting = particle[ weighting_ ];
                           // float_X const normedWeighting = weighting /
                            //    float_X( particles::TYPICAL_NUM_PARTICLES_PER_MACROPARTICLE );
                            float3_X const mom = particle[ momentum_ ] / weighting;
                            float3_X const pos = particle[ position_ ];
							
							lcellId_t const cellIdx = particle[ localCellIdx_ ];
							const DataSpace<simDim> frameCellOffset(DataSpaceOperations<simDim>::template map<MappingDesc::SuperCellSize > (cellIdx));
							const int index_y=frameCellOffset.y();
							
							auto globalCellOffset = globalOffset 
													+ (superCellIdx - mapper.getGuardingSuperCells()) * MappingDesc::SuperCellSize::toRT()
													+ frameCellOffset;	
							const float_X posX = ( float_X( globalCellOffset.x() ) + pos.x() ) * cellSize.x();
							
							if(	index_y >= 0 && index_y < SuperCellSize::y::value ){
								atomicAdd( &(shCount_e[index_y]), weighting, ::alpaka::hierarchy::Threads{});
								//weighted sum of single Electron values (Momentum = particle_momentum/normedWeighting)
								atomicAdd( &(shSumMom2[index_y]), mom.x() * mom.x() * weighting, ::alpaka::hierarchy::Threads{});
								atomicAdd( &(shSumPos2[index_y]), posX*posX*weighting, ::alpaka::hierarchy::Threads{});
								atomicAdd( &(shSumMomPos2[index_y]), mom.x()*posX* weighting, ::alpaka::hierarchy::Threads{});
							}
                        }
                    }
                );

                // set frame to next particle frame
                frame = pb.getPreviousFrame(frame);
                forEachParticle(
                    [&](
                        uint32_t const linearIdx,
                        uint32_t const idx
                    )
                    {
                        /* Update particle for the next round.
                         * The frame list is traverse from the last to the first frame.
                         * Only the last frame can contain gaps therefore all following
                         * frames are filled with fully particles.
                         */
                        currentParticleCtx[ idx ] = frame[ linearIdx ];
                    }
                );
            }


            // wait that all virtual threads updated the shared memory energies
            __syncthreads( );
			
			const int gOffset = (globalOffset + (superCellIdx - mapper.getGuardingSuperCells()) * MappingDesc::SuperCellSize::toRT()).y();
            ForEachIdx< SuperCellYDom >{ workerIdx }(
                [&](
                    uint32_t const linearIdx,
                    uint32_t const
				)
                {
					if(	gOffset >= 0 && gOffset < subGridY - SuperCellSize::y::value ){
						atomicAdd(&( gSumMom2[gOffset+linearIdx] ),static_cast< float_64 >( shSumMom2[linearIdx] ),::alpaka::hierarchy::Blocks{});
						atomicAdd(&( gSumPos2[gOffset+linearIdx] ),static_cast< float_64 >( shSumPos2[linearIdx] ),::alpaka::hierarchy::Blocks{});
						atomicAdd(&( gSumMomPos2[gOffset+linearIdx] ),static_cast< float_64 >( shSumMomPos2[linearIdx] ),::alpaka::hierarchy::Blocks{});
						atomicAdd(&( gCount_e[gOffset+linearIdx] ),static_cast< float_64 >( shCount_e[linearIdx] ),::alpaka::hierarchy::Blocks{});    
					}           
				}  
            );
            
        }
    };

    template< typename ParticlesType >
    class CalcEmittance : public plugins::multi::ISlave
    {
    public:

        struct Help : public plugins::multi::IHelp
        {

            /** creates an instance of ISlave
             *
             * @tparam T_Slave type of the interface implementation (must inherit from ISlave)
             * @param help plugin defined help
             * @param id index of the plugin, range: [0;help->getNumPlugins())
             */
            std::shared_ptr< ISlave > create(
                std::shared_ptr< IHelp > & help,
                size_t const id,
                MappingDesc* cellDescription
            )
            {
                return std::shared_ptr< ISlave >(
                    new CalcEmittance< ParticlesType >(
                        help,
                        id,
                        cellDescription
                    )
                );
            }

            // find all valid filter for the current used species
            using EligibleFilters = typename MakeSeqFromNestedSeq<
                typename bmpl::transform<
                    particles::filter::AllParticleFilters,
                    particles::traits::GenerateSolversIfSpeciesEligible<
                        bmpl::_1,
                        ParticlesType
                    >
                >::type
            >::type;

            //! periodicity of computing the particle energy
            plugins::multi::Option< std::string > notifyPeriod = {
                "period",
                "compute emittance[for each n-th step] enable plugin by setting a non-zero value"
            };
            plugins::multi::Option< std::string > filter = {
                "filter",
                "particle filter: "
            };

            //! string list with all possible particle filters
            std::string concatenatedFilterNames;
            std::vector< std::string > allowedFilters;

            ///! method used by plugin controller to get --help description
            void registerHelp(
                boost::program_options::options_description & desc,
                std::string const & masterPrefix = std::string{ }
            )
            {

                ForEach<
                    EligibleFilters,
                    plugins::misc::AppendName< bmpl::_1 >
                > getEligibleFilterNames;
                getEligibleFilterNames( forward( allowedFilters ) );

                concatenatedFilterNames = plugins::misc::concatenateToString(
                    allowedFilters,
                    ", "
                );

                notifyPeriod.registerHelp(
                    desc,
                    masterPrefix + prefix
                );
                filter.registerHelp(
                    desc,
                    masterPrefix + prefix,
                    std::string( "[" ) + concatenatedFilterNames + "]"
                );
            }

            void expandHelp(
                boost::program_options::options_description & desc,
                std::string const & masterPrefix = std::string{ }
            )
            {
            }


            void validateOptions()
            {
                if( notifyPeriod.size() != filter.size() )
                    throw std::runtime_error( name + ": parameter filter and period are not used the same number of times" );

                // check if user passed filter name are valid
                for( auto const & filterName : filter)
                {
                    if(
                        std::find(
                            allowedFilters.begin(),
                            allowedFilters.end(),
                            filterName
                        ) == allowedFilters.end()
                    )
                    {
                        throw std::runtime_error( name + ": unknown filter '" + filterName + "'" );
                    }
                }
            }

            size_t getNumPlugins() const
            {
                return notifyPeriod.size();
            }

            std::string getDescription() const
            {
                return description;
            }

            std::string getOptionPrefix() const
            {
                return prefix;
            }

            std::string getName() const
            {
                return name;
            }

            std::string const name = "CalcEmittance";
            //! short description of the plugin
            std::string const description = "calculate the emittance of a species";
            //! prefix used for command line arguments
            std::string const prefix = ParticlesType::FrameType::getName( ) + std::string( "_emittance" );
        };

        //! must be implemented by the user
        static std::shared_ptr< plugins::multi::IHelp > getHelp()
        {
            return std::shared_ptr< plugins::multi::IHelp >( new Help{ } );
        }

        CalcEmittance(
            std::shared_ptr< plugins::multi::IHelp > & help,
            size_t const id,
            MappingDesc* cellDescription
        ) :
            m_help( std::static_pointer_cast< Help >(help) ),
            m_id( id ),
            m_cellDescription( cellDescription )
        {
            filename = m_help->getOptionPrefix() + "_" + m_help->filter.get( m_id ) + ".dat";

            // decide which MPI-rank writes output
            writeToFile = reduce.hasResult( mpi::reduceMethods::Reduce( ) );
			const SubGrid<simDim>& subGrid = Environment<simDim>::get().SubGrid();
            // create 4 ints on gpu and host
            gSumMom2 = new GridBuffer<
                float_64,
                DIM1
            >( DataSpace< DIM1 >( subGrid.getGlobalDomain().size.y() ) );
            gSumPos2 = new GridBuffer<
                float_64,
                DIM1
            >( DataSpace< DIM1 >( subGrid.getGlobalDomain().size.y() ) );
            gSumMomPos2 = new GridBuffer<
                float_64,
                DIM1
            >( DataSpace< DIM1 >( subGrid.getGlobalDomain().size.y() ) );
            gCount_e = new GridBuffer<
                float_64,
                DIM1
            >( DataSpace< DIM1 >( subGrid.getGlobalDomain().size.y() ) );     

            // only MPI rank that writes to file
            if( writeToFile )
            {
                // open output file
                outFile.open(
                    filename.c_str( ),
                    std::ofstream::out | std::ostream::trunc
                );

                // error handling
                if( !outFile )
                {
                    std::cerr <<
                        "Can't open file [" <<
                        filename <<
                        "] for output, diasble plugin output. " <<
                        std::endl;
                    writeToFile = false;
                }

                // create header of the file
                outFile << "#step sum of x² sum of ux² sum of (x*ux)² number of electrons" << " \n";
            }

            // set how often the plugin should be executed while PIConGPU is running
            Environment<>::get( ).PluginConnector( ).setNotificationPeriod(
                this,
                m_help->notifyPeriod.get( id )
            );
        }

        virtual ~CalcEmittance( )
        {
            if( writeToFile )
            {
                outFile.flush( );
                // flush cached data to file
                outFile << std::endl;

                if( outFile.fail( ) )
                    std::cerr << "Error on flushing file [" << filename << "]. " << std::endl;
                outFile.close( );
            }
            // free global memory on GPU
            __delete( gSumMom2 );
            __delete( gSumPos2 );
            __delete( gSumMomPos2 );
            __delete( gCount_e );
        }

        /** this code is executed if the current time step is supposed to compute
         * gSumMom2, gSumPos2, gSumMomPos2, gCount_e
         */
        void notify( uint32_t currentStep )
        {
            // call the method that calls the plugin kernel
            calculateCalcEmittance < CORE + BORDER > ( currentStep );
        }


        void restart(
            uint32_t restartStep,
            std::string const & restartDirectory
        )
        {
            if( !writeToFile )
                return;

            writeToFile = restoreTxtFile(
                outFile,
                filename,
                restartStep,
                restartDirectory
            );
        }

        void checkpoint(
            uint32_t currentStep,
            std::string const & checkpointDirectory
        )
        {
            if( !writeToFile )
                return;

            checkpointTxtFile(
                outFile,
                filename,
                currentStep,
                checkpointDirectory
            );
        }
        

        
    private:
   
    
        //! method to call analysis and plugin-kernel calls
        template< uint32_t AREA >
        void calculateCalcEmittance( uint32_t currentStep )
        {
            DataConnector &dc = Environment<>::get( ).DataConnector( );

            // use data connector to get particle data
            auto particles = dc.get< ParticlesType >(
                ParticlesType::FrameType::getName( ),
                true
            );

            // initialize global gSum... with zero
            gSumMom2->getDeviceBuffer( ).setValue( 0.0 );
            gSumPos2->getDeviceBuffer( ).setValue( 0.0 );
            gSumMomPos2->getDeviceBuffer( ).setValue( 0.0 );
            gCount_e->getDeviceBuffer( ).setValue( 0.0 );

            constexpr uint32_t numWorkers = pmacc::traits::GetNumWorkers<
                pmacc::math::CT::volume< SuperCellSize >::type::value
            >::value;

            AreaMapping<
                AREA,
                MappingDesc
            > mapper( *m_cellDescription );

            auto kernel = PMACC_KERNEL( KernelCalcEmittance< numWorkers >{ } )(
                mapper.getGridDim( ),
                numWorkers
            );
            
              // Some funny things that make it possible for the kernel to calculate
			  // the absolute position of the particles
			DataSpace<simDim> localSize(m_cellDescription->getGridLayout().getDataSpaceWithoutGuarding());
			//const uint32_t numSlides = MovingWindow::getInstance().getSlideCounter(currentStep);
			const SubGrid<simDim>& subGrid = Environment<simDim>::get().SubGrid();
			const int subGridY = subGrid.getGlobalDomain().size.y();
			auto movingWindow=MovingWindow::getInstance().getWindow(currentStep);
			DataSpace<simDim> globalOffset(subGrid.getLocalDomain().offset);
			globalOffset.y() -= movingWindow.globalDimensions.offset.y();
			//globalOffset.y() += (localSize.y() * numSlides);
			
			std::cout << "debug emitt: \t" <<  currentStep << "\t" 
					  << movingWindow.globalDimensions.offset.y() << "\t"
					  << movingWindow.localDimensions.offset.y() << "\t"
					  << globalOffset.y() << std::endl;
            
            auto binaryKernel = std::bind(
                kernel,
                particles->getDeviceParticlesBox( ),
                gSumMom2->getDeviceBuffer( ).getDataBox( ),
                gSumPos2->getDeviceBuffer( ).getDataBox( ),
                gSumMomPos2->getDeviceBuffer( ).getDataBox( ),
                gCount_e->getDeviceBuffer( ).getDataBox( ),
                globalOffset,
                subGridY,
                mapper,
                std::placeholders::_1
            );

            ForEach<
                typename Help::EligibleFilters,
                plugins::misc::ExecuteIfNameIsEqual< bmpl::_1 >
            >{ }(
                m_help->filter.get( m_id ),
                currentStep,
                binaryKernel
            );

            dc.releaseData( ParticlesType::FrameType::getName( ) );

            // get gSum... from GPU
            gSumMom2->deviceToHost( );
            gSumPos2->deviceToHost( );
            gSumMomPos2->deviceToHost( );
            gCount_e->deviceToHost( );

            // create storage for the global reduced result          
			float_64 reducedSumMom2[ subGrid.getGlobalDomain().size.y()];
			float_64 reducedSumPos2[ subGrid.getGlobalDomain().size.y()];
			float_64 reducedSumMomPos2[ subGrid.getGlobalDomain().size.y()];
			float_64 reducedCount_e[ subGrid.getGlobalDomain().size.y()];
             

            // add gSum values from all GPUs using MPI
            reduce(
                nvidia::functors::Add( ),
                reducedSumMom2,
                gSumMom2->getHostBuffer( ).getBasePointer( ),
                subGrid.getGlobalDomain().size.y(),
                mpi::reduceMethods::Reduce( )
            );
            
            reduce(
                nvidia::functors::Add( ),
                reducedSumPos2,
                gSumPos2->getHostBuffer( ).getBasePointer( ),
                subGrid.getGlobalDomain().size.y(),
                mpi::reduceMethods::Reduce( )
            );
            
            reduce(
                nvidia::functors::Add( ),
                reducedSumMomPos2,
                gSumMomPos2->getHostBuffer( ).getBasePointer( ),
                subGrid.getGlobalDomain().size.y(),
                mpi::reduceMethods::Reduce( )
            );
            
            reduce(
                nvidia::functors::Add( ),
                reducedCount_e,
                gCount_e->getHostBuffer( ).getBasePointer( ),
                subGrid.getGlobalDomain().size.y(),
                mpi::reduceMethods::Reduce( )
            );

            /* print timestep, emittance to file: */
            if( writeToFile )
            {
				using dbl = std::numeric_limits< float_64 >;
				outFile.precision( dbl::digits10 );
				outFile << currentStep << " "
						<< std::scientific;
				for (int i=0; i<subGrid.getGlobalDomain().size.y();i++)
				{
					if (i % 10 == 0) {						
						double numElec = reducedCount_e[i];
						double mom2_SI = reducedSumMom2[i] / numElec * UNIT_MASS * UNIT_SPEED* UNIT_MASS * UNIT_SPEED;
						double pos2_SI = reducedSumPos2[i] *UNIT_LENGTH*UNIT_LENGTH / numElec;
						double mompos_SI = reducedSumMomPos2[i]*UNIT_MASS * UNIT_SPEED*UNIT_LENGTH / numElec;
						for (int j=i+1; j<i+9;j++){
							numElec += reducedCount_e[j];
							mom2_SI += reducedSumMom2[j] / numElec * UNIT_MASS * UNIT_SPEED* UNIT_MASS * UNIT_SPEED;
							pos2_SI += reducedSumPos2[j] *UNIT_LENGTH*UNIT_LENGTH / numElec;
							mompos_SI += reducedSumMomPos2[j]*UNIT_MASS * UNIT_SPEED*UNIT_LENGTH / numElec;
						} 
						double ux2=mom2_SI/(UNIT_SPEED*UNIT_SPEED*SI::ELECTRON_MASS_SI*SI::ELECTRON_MASS_SI);	
						double xux =mompos_SI/(UNIT_SPEED*SI::ELECTRON_MASS_SI);
						outFile << algorithms::math::sqrt((pos2_SI * ux2 - xux * xux)) << " "; 
					}
				}   
				outFile << std::endl;               
            }
        }

        //! energy values (global on GPU)
        GridBuffer<
            float_64,
            DIM1
        > * gSumMom2 = nullptr;
        
        GridBuffer<
            float_64,
            DIM1
        > * gSumPos2 = nullptr;
        
        GridBuffer<
            float_64,
            DIM1
        > * gSumMomPos2 = nullptr;
        
        GridBuffer<
            float_64,
            DIM1
        > * gCount_e = nullptr;

        MappingDesc* m_cellDescription;

        //! output file name
        std::string filename;

        //! file output stream
        std::ofstream outFile;

        /** only one MPI rank creates a file
         *
         * true if this MPI rank creates the file, else false
         */
        bool writeToFile = false;

        //! MPI reduce to add all energies over several GPUs
        mpi::MPIReduce reduce;

        std::shared_ptr< Help > m_help;
        size_t m_id;
    };

namespace particles
{
namespace traits
{
    template<
        typename T_Species,
        typename T_UnspecifiedSpecies
    >
    struct SpeciesEligibleForSolver<
        T_Species,
        CalcEmittance< T_UnspecifiedSpecies >
    >
    {
        using FrameType = typename T_Species::FrameType;

        // this plugin needs at least the weighting and momentum attributes
        using RequiredIdentifiers = MakeSeq_t<
            weighting,
            momentum
        >;

        using SpeciesHasIdentifiers = typename pmacc::traits::HasIdentifiers<
            FrameType,
            RequiredIdentifiers
        >::type;

        // and also a mass ratio for energy calculation from momentum
        using SpeciesHasFlags = typename pmacc::traits::HasFlag<
            FrameType,
            massRatio<>
        >::type;

        using type = typename bmpl::and_<
            SpeciesHasIdentifiers,
            SpeciesHasFlags
        >;
    };
} // namespace traits
} // namespace particles
} // namespace picongpu
