/**
* This file is part of ManhattanSLAM.
*
* Copyright (C) 2021 Raza Yunus <razayunus31 at gmail dot com>
* For more information see <https://github.com/razayunus/ManhattanSLAM>
*
* ManhattanSLAM is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ManhattanSLAM is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ManhattanSLAM. If not, see <http://www.gnu.org/licenses/>.
*/

/**
 * This code is adopted and modified from https://github.com/HKUST-Aerial-Robotics/DenseSurfelMapping.
 */

#ifndef SURFEL_H
#define SURFEL_H

struct Surfel {
    float px, py, pz;
    float nx, ny, nz;
    float size;
    float color;
    int r, g, b;
    float weight;
    int updateTimes;
    int lastUpdate;
};

#endif //SURFEL_H
