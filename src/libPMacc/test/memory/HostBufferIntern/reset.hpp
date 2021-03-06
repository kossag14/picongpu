/* Copyright 2015-2017 Erik Zenker
 *
 * This file is part of libPMacc.
 *
 * libPMacc is free software: you can redistribute it and/or modify
 * it under the terms of either the GNU General Public License or
 * the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * libPMacc is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License and the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * and the GNU Lesser General Public License along with libPMacc.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

/* #includes in "test/memoryUT.cu" */

/**
 * Checks if the HostBufferIntern is reseted correctly to zero.
 */
struct ResetTest {

    template<typename T_Dim>
    void exec(T_Dim)
    {

        typedef uint8_t Data;
        typedef size_t Extents;

        std::vector<size_t> nElementsPerDim = getElementsPerDim<T_Dim>();

        for(unsigned i = 0; i < nElementsPerDim.size(); ++i)
        {
            ::PMacc::DataSpace<T_Dim::value> const dataSpace = ::PMacc::DataSpace<T_Dim::value>::create(nElementsPerDim[i]);
            ::PMacc::HostBufferIntern<Data, T_Dim::value> hostBufferIntern(dataSpace);

            hostBufferIntern.reset();

            for(size_t i = 0; i < static_cast<size_t>(dataSpace.productOfComponents()); ++i){
                BOOST_CHECK_EQUAL( hostBufferIntern.getPointer()[i], 0 );
            }

        }

    }

    PMACC_NO_NVCC_HDWARNING
    template<typename T_Dim>
    HDINLINE void operator()(T_Dim dim)
    {
        exec(dim);
    }
};

BOOST_AUTO_TEST_CASE( reset ){
    ::boost::mpl::for_each< Dims >( ResetTest() );

}
